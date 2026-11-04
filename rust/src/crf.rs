//! Linear-chain CRF tagger over hashed token features, with BIO tags.
//!
//! Training minimises the negative log-likelihood with AdaGrad. Forward-backward runs in scaled
//! probability space (no `exp` in the inner loop); decoding uses Viterbi in log space and
//! reports each span's mean token marginal as its score.

// Index loops mirror the forward-backward equations; iterator chains would obscure them.
#![allow(clippy::needless_range_loop)]

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::finding::{bytes_to_chars, Finding, NER_RANK};
use crate::tokenize::{is_space, tokenize};

/// A character-offset span `(start, end, label)`.
pub type Span = (usize, usize, String);

/// Token byte offsets and gold tag ids of one training sequence.
type Tagged = (Vec<(usize, usize)>, Vec<u8>);

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const MAGIC: &[u8; 8] = b"DLPCRF01";
const SHAPE_LIMIT: usize = 16;
/// Number of features emitted per token.
pub const FEATURES_PER_TOKEN: usize = 19;

fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn feature(template: &str, parts: &[&str]) -> u64 {
    let mut h = fnv(FNV_OFFSET, template.as_bytes());
    h = fnv(h, b"=");
    for p in parts {
        h = fnv(h, p.as_bytes());
    }
    h
}

fn shape(word: &str) -> String {
    word.chars()
        .map(|c| {
            if c.is_uppercase() {
                'X'
            } else if c.is_lowercase() {
                'x'
            } else if c.is_numeric() {
                'd'
            } else {
                c
            }
        })
        .collect()
}

fn short_shape(full: &str) -> String {
    let mut out = String::new();
    let mut last: Option<char> = None;
    for c in full.chars() {
        if last != Some(c) {
            out.push(c);
            last = Some(c);
        }
    }
    out.chars().take(SHAPE_LIMIT).collect()
}

fn head(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

fn tail(s: &str, n: usize) -> &str {
    let count = s.chars().count();
    if count <= n {
        return s;
    }
    let (i, _) = s.char_indices().nth(count - n).expect("in range");
    &s[i..]
}

/// Hashed feature values (before reduction modulo the bucket count) for each token.
pub fn token_features(text: &str, tokens: &[(usize, usize)]) -> Vec<[u64; FEATURES_PER_TOKEN]> {
    let n = tokens.len();
    let lower: Vec<String> = tokens
        .iter()
        .map(|&(s, e)| text[s..e].to_lowercase())
        .collect();
    let shapes: Vec<String> = tokens.iter().map(|&(s, e)| shape(&text[s..e])).collect();
    let shorts: Vec<String> = shapes.iter().map(|s| short_shape(s)).collect();
    let gap = |i: usize| -> &'static str {
        // Whitespace between token i-1 and token i.
        if i == 0 {
            "s"
        } else if i == n {
            "e"
        } else if tokens[i - 1].1 < tokens[i].0 {
            "1"
        } else {
            "0"
        }
    };
    let word = |i: isize| -> &str {
        if i < 0 {
            "<s>"
        } else if i as usize >= n {
            "</s>"
        } else {
            &lower[i as usize]
        }
    };
    let short = |i: isize| -> &str {
        if i < 0 {
            "<s>"
        } else if i as usize >= n {
            "</s>"
        } else {
            &shorts[i as usize]
        }
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let w = &lower[i];
        let ii = i as isize;
        let nchars = w.chars().count().min(12).to_string();
        let full_shape: String = shapes[i].chars().take(SHAPE_LIMIT).collect();
        out.push([
            fnv(FNV_OFFSET, b"b"),
            feature("w", &[w]),
            feature("p2", &[head(w, 2)]),
            feature("p3", &[head(w, 3)]),
            feature("s2", &[tail(w, 2)]),
            feature("s3", &[tail(w, 3)]),
            feature("sh", &[&full_shape]),
            feature("ss", &[&shorts[i]]),
            feature("ln", &[&nchars]),
            feature("g", &[gap(i)]),
            feature("ga", &[gap(i + 1)]),
            feature("w-1", &[word(ii - 1)]),
            feature("w+1", &[word(ii + 1)]),
            feature("w-2", &[word(ii - 2)]),
            feature("w+2", &[word(ii + 2)]),
            feature("ss-1", &[short(ii - 1)]),
            feature("ss+1", &[short(ii + 1)]),
            feature("w-1w", &[word(ii - 1), "|", w]),
            feature(
                "ss-1ss",
                &[short(ii - 1), "|", &shorts[i], "|", short(ii + 1)],
            ),
        ]);
    }
    out
}

/// BIO tag names for entity `labels`: `O`, then `B-x`, `I-x` per label.
pub fn tag_names(labels: &[String]) -> Vec<String> {
    let mut tags = vec!["O".to_string()];
    for l in labels {
        tags.push(format!("B-{l}"));
        tags.push(format!("I-{l}"));
    }
    tags
}

/// Gold BIO tags for `tokens` (byte offsets) from character-offset spans. A token belongs to
/// the first span (in start order) that it overlaps; labels not in `labels` are ignored.
pub fn gold_tags(
    text: &str,
    tokens: &[(usize, usize)],
    spans: &[Span],
    labels: &[String],
) -> Vec<u8> {
    let mut offsets: Vec<usize> = tokens.iter().flat_map(|&(s, e)| [s, e]).collect();
    bytes_to_chars(text, &mut offsets);
    let mut tags = vec![0u8; tokens.len()];
    let mut sorted: Vec<&Span> = spans.iter().collect();
    sorted.sort_by_key(|s| (s.0, s.1));
    for (start, end, label) in sorted {
        let Some(li) = labels.iter().position(|l| l == label) else {
            continue;
        };
        let mut first = true;
        for (t, tag) in tags.iter_mut().enumerate() {
            let (ts, te) = (offsets[2 * t], offsets[2 * t + 1]);
            if ts < *end && te > *start && *tag == 0 {
                *tag = if first {
                    1 + 2 * li as u8
                } else {
                    2 + 2 * li as u8
                };
                first = false;
            }
        }
    }
    tags
}

#[derive(Serialize, Deserialize)]
struct Header {
    format: String,
    version: u32,
    labels: Vec<String>,
    n_buckets: usize,
}

/// Training settings.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub n_buckets: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub seed: u64,
    pub verbose: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            n_buckets: 1 << 20,
            epochs: 3,
            learning_rate: 0.1,
            batch_size: 64,
            seed: 0,
            verbose: false,
        }
    }
}

/// A trained (or empty) tagger.
pub struct Crf {
    pub labels: Vec<String>,
    pub tags: Vec<String>,
    pub n_buckets: usize,
    /// Emission weights, `n_buckets x tags` row-major.
    pub weights: Vec<f32>,
    /// Transition weights `from x to`.
    pub transitions: Vec<f32>,
    pub start: Vec<f32>,
    pub end: Vec<f32>,
}

struct Lattice {
    n: usize,
    t: usize,
    /// Emission scores, n x t.
    emit: Vec<f64>,
}

struct Marginals {
    log_z: f64,
    /// Token marginals, n x t.
    unary: Vec<f64>,
    /// Summed pairwise marginals, t x t.
    pairwise: Vec<f64>,
}

struct ExpParams {
    trans: Vec<f64>,
    start: Vec<f64>,
    end: Vec<f64>,
}

struct Grad {
    nll: f64,
    feats: Vec<u32>,
    /// Per token: (tag, gradient) pairs with a non-negligible gradient.
    token_grads: Vec<Vec<(u8, f32)>>,
    trans: Vec<f64>,
    start: Vec<f64>,
    end: Vec<f64>,
}

impl Crf {
    pub fn new(labels: Vec<String>, n_buckets: usize) -> Crf {
        let tags = tag_names(&labels);
        let t = tags.len();
        Crf {
            labels,
            tags,
            n_buckets,
            weights: vec![0.0; n_buckets * t],
            transitions: vec![0.0; t * t],
            start: vec![0.0; t],
            end: vec![0.0; t],
        }
    }

    pub fn num_tags(&self) -> usize {
        self.tags.len()
    }

    /// Feature buckets per token, flattened (`FEATURES_PER_TOKEN` per token).
    pub fn feature_ids(&self, text: &str, tokens: &[(usize, usize)]) -> Vec<u32> {
        token_features(text, tokens)
            .iter()
            .flat_map(|fs| fs.iter().map(|h| (h % self.n_buckets as u64) as u32))
            .collect()
    }

    fn lattice(&self, feats: &[u32], n: usize) -> Lattice {
        let t = self.num_tags();
        let mut emit = vec![0.0f64; n * t];
        for i in 0..n {
            let row = &mut emit[i * t..(i + 1) * t];
            for &f in &feats[i * FEATURES_PER_TOKEN..(i + 1) * FEATURES_PER_TOKEN] {
                let w = &self.weights[f as usize * t..(f as usize + 1) * t];
                for (r, &x) in row.iter_mut().zip(w) {
                    *r += x as f64;
                }
            }
        }
        Lattice { n, t, emit }
    }

    fn exp_params(&self) -> ExpParams {
        ExpParams {
            trans: self.transitions.iter().map(|&x| (x as f64).exp()).collect(),
            start: self.start.iter().map(|&x| (x as f64).exp()).collect(),
            end: self.end.iter().map(|&x| (x as f64).exp()).collect(),
        }
    }

    /// Scaled forward-backward.
    fn marginals(&self, lat: &Lattice, ep: &ExpParams, want_pairwise: bool) -> Marginals {
        let (n, t) = (lat.n, lat.t);
        let mut psi = vec![0.0f64; n * t];
        let mut log_z = 0.0;
        for i in 0..n {
            let row = &lat.emit[i * t..(i + 1) * t];
            let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            log_z += m;
            for y in 0..t {
                psi[i * t + y] = (row[y] - m).exp();
            }
        }
        let mut alpha = vec![0.0f64; n * t];
        let mut scale = vec![0.0f64; n];
        for y in 0..t {
            alpha[y] = ep.start[y] * psi[y];
        }
        scale[0] = alpha[..t].iter().sum();
        for a in &mut alpha[..t] {
            *a /= scale[0];
        }
        for i in 1..n {
            let (prev, cur) = alpha.split_at_mut(i * t);
            let prev = &prev[(i - 1) * t..];
            let cur = &mut cur[..t];
            for (yp, &a) in prev.iter().enumerate() {
                if a == 0.0 {
                    continue;
                }
                let tr = &ep.trans[yp * t..(yp + 1) * t];
                for y in 0..t {
                    cur[y] += a * tr[y];
                }
            }
            let mut s = 0.0;
            for y in 0..t {
                cur[y] *= psi[i * t + y];
                s += cur[y];
            }
            scale[i] = s;
            for c in cur.iter_mut() {
                *c /= s;
            }
        }
        let z_end: f64 = (0..t).map(|y| alpha[(n - 1) * t + y] * ep.end[y]).sum();
        log_z += scale.iter().map(|s| s.ln()).sum::<f64>() + z_end.ln();
        let mut beta = vec![0.0f64; n * t];
        beta[(n - 1) * t..].copy_from_slice(&ep.end);
        let mut tmp = vec![0.0f64; t];
        for i in (0..n - 1).rev() {
            for y in 0..t {
                tmp[y] = psi[(i + 1) * t + y] * beta[(i + 1) * t + y];
            }
            for y in 0..t {
                let tr = &ep.trans[y * t..(y + 1) * t];
                let s: f64 = tr.iter().zip(&tmp).map(|(a, b)| a * b).sum();
                beta[i * t + y] = s / scale[i + 1];
            }
        }
        let mut unary = vec![0.0f64; n * t];
        for k in 0..n * t {
            unary[k] = alpha[k] * beta[k] / z_end;
        }
        let mut pairwise = Vec::new();
        if want_pairwise {
            pairwise = vec![0.0f64; t * t];
            for i in 1..n {
                let denom = scale[i] * z_end;
                for y in 0..t {
                    tmp[y] = psi[i * t + y] * beta[i * t + y] / denom;
                }
                for yp in 0..t {
                    let a = alpha[(i - 1) * t + yp];
                    if a == 0.0 {
                        continue;
                    }
                    let tr = &ep.trans[yp * t..(yp + 1) * t];
                    let row = &mut pairwise[yp * t..(yp + 1) * t];
                    for y in 0..t {
                        row[y] += a * tr[y] * tmp[y];
                    }
                }
            }
        }
        Marginals {
            log_z,
            unary,
            pairwise,
        }
    }

    fn gold_score(&self, lat: &Lattice, gold: &[u8]) -> f64 {
        let t = lat.t;
        let mut s = self.start[gold[0] as usize] as f64 + self.end[gold[lat.n - 1] as usize] as f64;
        for (i, &g) in gold.iter().enumerate() {
            s += lat.emit[i * t + g as usize];
            if i > 0 {
                s += self.transitions[gold[i - 1] as usize * t + g as usize] as f64;
            }
        }
        s
    }

    /// Negative log-likelihood of `gold` tags (used by tests and training diagnostics).
    pub fn nll(&self, text: &str, gold: &[u8]) -> f64 {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return 0.0;
        }
        let feats = self.feature_ids(text, &tokens);
        let lat = self.lattice(&feats, tokens.len());
        let m = self.marginals(&lat, &self.exp_params(), false);
        m.log_z - self.gold_score(&lat, gold)
    }

    fn gradient(&self, text: &str, tokens: &[(usize, usize)], gold: &[u8], ep: &ExpParams) -> Grad {
        let t = self.num_tags();
        let n = tokens.len();
        let feats = self.feature_ids(text, tokens);
        let lat = self.lattice(&feats, n);
        let m = self.marginals(&lat, ep, true);
        let nll = m.log_z - self.gold_score(&lat, gold);
        let mut token_grads = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::new();
            for y in 0..t {
                let g = m.unary[i * t + y] - if gold[i] as usize == y { 1.0 } else { 0.0 };
                if g.abs() > 1e-4 {
                    row.push((y as u8, g as f32));
                }
            }
            token_grads.push(row);
        }
        let mut trans = m.pairwise;
        for i in 1..n {
            trans[gold[i - 1] as usize * t + gold[i] as usize] -= 1.0;
        }
        let mut start: Vec<f64> = m.unary[..t].to_vec();
        start[gold[0] as usize] -= 1.0;
        let mut end: Vec<f64> = m.unary[(n - 1) * t..].to_vec();
        end[gold[n - 1] as usize] -= 1.0;
        Grad {
            nll,
            feats,
            token_grads,
            trans,
            start,
            end,
        }
    }

    /// Train on `texts` with character-offset `spans` (`(start, end, label)`).
    pub fn train(
        texts: &[String],
        spans: &[Vec<Span>],
        labels: Vec<String>,
        cfg: &TrainConfig,
    ) -> Crf {
        let mut model = Crf::new(labels, cfg.n_buckets);
        let t = model.num_tags();
        let data: Vec<Tagged> = texts
            .par_iter()
            .zip(spans.par_iter())
            .map(|(text, sp)| {
                let toks = tokenize(text);
                let gold = gold_tags(text, &toks, sp, &model.labels);
                (toks, gold)
            })
            .collect();
        let mut order: Vec<usize> = (0..texts.len())
            .filter(|&i| !data[i].0.is_empty())
            .collect();
        let mut acc_w = vec![0.0f32; model.weights.len()];
        let mut acc_trans = vec![0.0f64; t * t];
        let mut acc_start = vec![0.0f64; t];
        let mut acc_end = vec![0.0f64; t];
        let lr = cfg.learning_rate;
        let eps = 1e-8;
        let mut rng = cfg.seed ^ 0x9E37_79B9_7F4A_7C15;
        for epoch in 0..cfg.epochs {
            // Fisher-Yates with xorshift64*.
            for i in (1..order.len()).rev() {
                rng ^= rng >> 12;
                rng ^= rng << 25;
                rng ^= rng >> 27;
                let r = rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
                order.swap(i, (r % (i as u64 + 1)) as usize);
            }
            let mut total_nll = 0.0;
            for batch in order.chunks(cfg.batch_size.max(1)) {
                let ep = model.exp_params();
                let grads: Vec<Grad> = batch
                    .par_iter()
                    .map(|&i| model.gradient(&texts[i], &data[i].0, &data[i].1, &ep))
                    .collect();
                let mut d_trans = vec![0.0f64; t * t];
                let mut d_start = vec![0.0f64; t];
                let mut d_end = vec![0.0f64; t];
                for g in &grads {
                    total_nll += g.nll;
                    for (i, row) in g.token_grads.iter().enumerate() {
                        for &f in &g.feats[i * FEATURES_PER_TOKEN..(i + 1) * FEATURES_PER_TOKEN] {
                            let base = f as usize * t;
                            for &(y, gv) in row {
                                let k = base + y as usize;
                                acc_w[k] += gv * gv;
                                model.weights[k] -=
                                    (lr as f32) * gv / (acc_w[k].sqrt() + eps as f32);
                            }
                        }
                    }
                    for k in 0..t * t {
                        d_trans[k] += g.trans[k];
                    }
                    for k in 0..t {
                        d_start[k] += g.start[k];
                        d_end[k] += g.end[k];
                    }
                }
                let step = |w: &mut [f32], acc: &mut [f64], d: &[f64]| {
                    for k in 0..w.len() {
                        acc[k] += d[k] * d[k];
                        w[k] -= (lr * d[k] / (acc[k].sqrt() + eps)) as f32;
                    }
                };
                step(&mut model.transitions, &mut acc_trans, &d_trans);
                step(&mut model.start, &mut acc_start, &d_start);
                step(&mut model.end, &mut acc_end, &d_end);
            }
            if cfg.verbose {
                eprintln!(
                    "crf epoch {}: mean nll {:.4} over {} sequences",
                    epoch + 1,
                    total_nll / order.len().max(1) as f64,
                    order.len()
                );
            }
        }
        model
    }

    /// Viterbi path.
    fn viterbi(&self, lat: &Lattice) -> Vec<usize> {
        let (n, t) = (lat.n, lat.t);
        let mut score: Vec<f64> = (0..t).map(|y| self.start[y] as f64 + lat.emit[y]).collect();
        let mut back = vec![0usize; n * t];
        let mut next = vec![0.0f64; t];
        for i in 1..n {
            for y in 0..t {
                let mut best = f64::NEG_INFINITY;
                let mut arg = 0;
                for (yp, &s) in score.iter().enumerate() {
                    let v = s + self.transitions[yp * t + y] as f64;
                    if v > best {
                        best = v;
                        arg = yp;
                    }
                }
                next[y] = best + lat.emit[i * t + y];
                back[i * t + y] = arg;
            }
            std::mem::swap(&mut score, &mut next);
        }
        let mut best = f64::NEG_INFINITY;
        let mut arg = 0;
        for (y, &s) in score.iter().enumerate() {
            let v = s + self.end[y] as f64;
            if v > best {
                best = v;
                arg = y;
            }
        }
        let mut path = vec![0usize; n];
        path[n - 1] = arg;
        for i in (1..n).rev() {
            path[i - 1] = back[i * t + path[i]];
        }
        path
    }

    /// Tag ids of the best path (for tests).
    pub fn tag_path(&self, text: &str) -> Vec<usize> {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return Vec::new();
        }
        let feats = self.feature_ids(text, &tokens);
        self.viterbi(&self.lattice(&feats, tokens.len()))
    }

    /// Entity spans (character offsets) with their mean token marginal as score.
    pub fn predict(&self, text: &str) -> Vec<Finding> {
        let tokens = tokenize(text);
        if tokens.is_empty() {
            return Vec::new();
        }
        let n = tokens.len();
        let t = self.num_tags();
        let feats = self.feature_ids(text, &tokens);
        let lat = self.lattice(&feats, n);
        let path = self.viterbi(&lat);
        let marg = self.marginals(&lat, &self.exp_params(), false);
        let mut spans: Vec<(usize, usize, usize, f64)> = Vec::new(); // tok start, tok end, label, sum
        for (i, &tag) in path.iter().enumerate() {
            if tag == 0 {
                continue;
            }
            let label = (tag - 1) / 2;
            let inside = tag % 2 == 0;
            let p = marg.unary[i * t + tag];
            match spans.last_mut() {
                Some(last) if inside && last.1 == i && last.2 == label => {
                    last.1 = i + 1;
                    last.3 += p;
                }
                _ => spans.push((i, i + 1, label, p)),
            }
        }
        let mut offsets: Vec<usize> = spans
            .iter()
            .flat_map(|s| [tokens[s.0].0, tokens[s.1 - 1].1])
            .collect();
        bytes_to_chars(text, &mut offsets);
        spans
            .iter()
            .enumerate()
            .map(|(k, s)| Finding {
                start: offsets[2 * k],
                end: offsets[2 * k + 1],
                match_start: offsets[2 * k],
                match_end: offsets[2 * k + 1],
                label: self.labels[s.2].clone(),
                source: "ner".to_string(),
                score: s.3 / (s.1 - s.0) as f64,
                rank: NER_RANK,
            })
            .collect()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut out = BufWriter::new(File::create(path)?);
        let header = serde_json::to_vec(&Header {
            format: "llm-dlp-guard-crf".into(),
            version: 1,
            labels: self.labels.clone(),
            n_buckets: self.n_buckets,
        })?;
        out.write_all(MAGIC)?;
        out.write_all(&(header.len() as u32).to_le_bytes())?;
        out.write_all(&header)?;
        for v in self.start.iter().chain(&self.end).chain(&self.transitions) {
            out.write_all(&v.to_le_bytes())?;
        }
        let t = self.num_tags();
        let rows: Vec<usize> = (0..self.n_buckets)
            .filter(|&r| self.weights[r * t..(r + 1) * t].iter().any(|&x| x != 0.0))
            .collect();
        out.write_all(&(rows.len() as u64).to_le_bytes())?;
        for r in rows {
            out.write_all(&(r as u32).to_le_bytes())?;
            for v in &self.weights[r * t..(r + 1) * t] {
                out.write_all(&v.to_le_bytes())?;
            }
        }
        out.flush()
    }

    pub fn load(path: &str) -> std::io::Result<Crf> {
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        let mut input = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad("not an llm-dlp-guard CRF model"));
        }
        let mut u32b = [0u8; 4];
        input.read_exact(&mut u32b)?;
        let mut header = vec![0u8; u32::from_le_bytes(u32b) as usize];
        input.read_exact(&mut header)?;
        let header: Header = serde_json::from_slice(&header).map_err(|e| bad(&e.to_string()))?;
        if header.n_buckets == 0 || header.n_buckets > (1 << 26) {
            return Err(bad("invalid bucket count"));
        }
        let mut model = Crf::new(header.labels, header.n_buckets);
        let t = model.num_tags();
        fn read_f32(input: &mut BufReader<File>) -> std::io::Result<f32> {
            let mut b = [0u8; 4];
            input.read_exact(&mut b)?;
            Ok(f32::from_le_bytes(b))
        }
        for k in 0..t {
            model.start[k] = read_f32(&mut input)?;
        }
        for k in 0..t {
            model.end[k] = read_f32(&mut input)?;
        }
        for k in 0..t * t {
            model.transitions[k] = read_f32(&mut input)?;
        }
        let mut u64b = [0u8; 8];
        input.read_exact(&mut u64b)?;
        let rows = u64::from_le_bytes(u64b) as usize;
        if rows > model.n_buckets {
            return Err(bad("too many rows"));
        }
        let mut row = vec![0u8; 4 * t];
        for _ in 0..rows {
            input.read_exact(&mut u32b)?;
            let r = u32::from_le_bytes(u32b) as usize;
            if r >= model.n_buckets {
                return Err(bad("row out of range"));
            }
            input.read_exact(&mut row)?;
            for k in 0..t {
                let b = [row[4 * k], row[4 * k + 1], row[4 * k + 2], row[4 * k + 3]];
                model.weights[r * t + k] = f32::from_le_bytes(b);
            }
        }
        Ok(model)
    }
}

/// True if `text` has no tokens (only whitespace).
pub fn is_blank(text: &str) -> bool {
    text.chars().all(is_space)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> (Vec<String>, Vec<Vec<Span>>) {
        let mut texts = Vec::new();
        let mut spans = Vec::new();
        let names = [
            "Alice Moreau",
            "Bram Okafor",
            "Chen Liang",
            "Dana Ruiz",
            "Emil Varga",
        ];
        let cities = ["Lyon", "Utrecht", "Porto", "Graz", "Turku"];
        for (i, name) in names.iter().enumerate() {
            for city in cities.iter() {
                let text = format!("Please ask {name} to visit {city} next week.");
                let s = 11;
                let e = s + name.chars().count();
                let cs = e + 10;
                let ce = cs + city.chars().count();
                texts.push(text);
                spans.push(vec![
                    (s, e, "NAME".to_string()),
                    (cs, ce, "CITY".to_string()),
                ]);
                let _ = i;
            }
        }
        (texts, spans)
    }

    #[test]
    fn learns_toy_task_and_round_trips() {
        let (texts, spans) = toy();
        let cfg = TrainConfig {
            n_buckets: 1 << 12,
            epochs: 8,
            learning_rate: 0.2,
            batch_size: 4,
            seed: 1,
            verbose: false,
        };
        let model = Crf::train(&texts, &spans, vec!["NAME".into(), "CITY".into()], &cfg);
        let found = model.predict("Please ask Emil Varga to visit Porto next week.");
        let got: Vec<(usize, usize, &str)> = found
            .iter()
            .map(|f| (f.start, f.end, f.label.as_str()))
            .collect();
        assert_eq!(got, vec![(11, 21, "NAME"), (31, 36, "CITY")]);
        assert!(found.iter().all(|f| f.score > 0.5 && f.score <= 1.0 + 1e-9));
        let dir = std::env::temp_dir().join(format!("dlp-crf-{}", std::process::id()));
        let path = dir.to_str().unwrap().to_string();
        model.save(&path).unwrap();
        let loaded = Crf::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.weights, model.weights);
        assert_eq!(
            loaded.predict("Please ask Emil Varga to visit Porto next week."),
            found
        );
        assert!(model.predict("   ").is_empty());
        assert!(is_blank(" \n"));
    }

    #[test]
    fn marginals_sum_to_one_and_nll_positive() {
        let (texts, spans) = toy();
        let cfg = TrainConfig {
            n_buckets: 1 << 10,
            epochs: 1,
            ..TrainConfig::default()
        };
        let model = Crf::train(
            &texts[..5],
            &spans[..5],
            vec!["NAME".into(), "CITY".into()],
            &cfg,
        );
        let toks = tokenize(&texts[0]);
        let feats = model.feature_ids(&texts[0], &toks);
        let lat = model.lattice(&feats, toks.len());
        let m = model.marginals(&lat, &model.exp_params(), true);
        let t = model.num_tags();
        for i in 0..toks.len() {
            let s: f64 = m.unary[i * t..(i + 1) * t].iter().sum();
            assert!((s - 1.0).abs() < 1e-9);
        }
        let pair_total: f64 = m.pairwise.iter().sum();
        assert!((pair_total - (toks.len() - 1) as f64).abs() < 1e-6);
        let gold = gold_tags(&texts[0], &toks, &spans[0], &model.labels);
        assert!(model.nll(&texts[0], &gold) > 0.0);
    }

    #[test]
    fn gold_tags_bio() {
        let text = "Zoë Ng lives";
        let toks = tokenize(text);
        let tags = gold_tags(text, &toks, &[(0, 6, "NAME".into())], &["NAME".into()]);
        assert_eq!(tags, vec![1, 2, 0]);
        assert_eq!(tag_names(&["A".into()]), vec!["O", "B-A", "I-A"]);
    }
}
