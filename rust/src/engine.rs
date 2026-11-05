//! The pattern engine: an Aho-Corasick pass over every pattern's literal anchors decides which
//! regular expressions run; matches are validated, scored and resolved to non-overlapping spans.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use regex::Regex;
use serde::Deserialize;

use crate::finding::{bytes_to_chars, char_to_byte, resolve, Finding};
use crate::validators;

/// One catalog entry, as serialized by the Python catalog.
#[derive(Clone, Debug, Deserialize)]
pub struct PatternSpec {
    pub name: String,
    pub label: String,
    pub regex: String,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub anchors: Vec<String>,
    #[serde(default)]
    pub validator: Option<String>,
    pub score: f64,
}

#[derive(Debug, Deserialize)]
struct Catalog {
    patterns: Vec<PatternSpec>,
}

struct Compiled {
    spec: PatternSpec,
    regex: Regex,
    group: Option<usize>,
}

/// Compiled catalog.
pub struct Engine {
    patterns: Vec<Compiled>,
    prefilter: Option<AhoCorasick>,
    anchor_patterns: Vec<Vec<usize>>,
    always: Vec<usize>,
}

impl Engine {
    /// Build from catalog JSON: `{"patterns": [{name, label, regex, group?, anchors?, validator?, score}]}`.
    pub fn from_json(json: &str) -> Result<Engine, String> {
        let catalog: Catalog = serde_json::from_str(json).map_err(|e| e.to_string())?;
        Engine::new(catalog.patterns)
    }

    pub fn new(specs: Vec<PatternSpec>) -> Result<Engine, String> {
        let mut patterns = Vec::with_capacity(specs.len());
        let mut anchors: Vec<String> = Vec::new();
        let mut anchor_patterns: Vec<Vec<usize>> = Vec::new();
        let mut always = Vec::new();
        for (idx, spec) in specs.into_iter().enumerate() {
            let regex = Regex::new(&spec.regex).map_err(|e| format!("{}: {e}", spec.name))?;
            let group = match &spec.group {
                Some(name) => Some(
                    regex
                        .capture_names()
                        .position(|n| n == Some(name.as_str()))
                        .ok_or_else(|| format!("{}: no group named {name}", spec.name))?,
                ),
                None => None,
            };
            if let Some(v) = &spec.validator {
                if !validators::VALIDATORS.contains(&v.as_str()) {
                    return Err(format!("{}: unknown validator {v}", spec.name));
                }
            }
            if !(0.0..=1.0).contains(&spec.score) {
                return Err(format!("{}: score must be in [0, 1]", spec.name));
            }
            if spec.anchors.is_empty() {
                always.push(idx);
            }
            for anchor in &spec.anchors {
                if anchor.is_empty() {
                    return Err(format!("{}: empty anchor", spec.name));
                }
                let lower = anchor.to_ascii_lowercase();
                match anchors.iter().position(|a| *a == lower) {
                    Some(pos) => anchor_patterns[pos].push(idx),
                    None => {
                        anchors.push(lower);
                        anchor_patterns.push(vec![idx]);
                    }
                }
            }
            patterns.push(Compiled { spec, regex, group });
        }
        let prefilter = if anchors.is_empty() {
            None
        } else {
            Some(
                AhoCorasickBuilder::new()
                    .ascii_case_insensitive(true)
                    .match_kind(MatchKind::Standard)
                    .build(&anchors)
                    .map_err(|e| e.to_string())?,
            )
        };
        Ok(Engine {
            patterns,
            prefilter,
            anchor_patterns,
            always,
        })
    }

    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Indices of the patterns whose anchors occur in `text`, plus the anchorless ones.
    pub fn active(&self, text: &str) -> Vec<usize> {
        let mut active = vec![false; self.patterns.len()];
        for &i in &self.always {
            active[i] = true;
        }
        if let Some(ac) = &self.prefilter {
            for m in ac.find_overlapping_iter(text) {
                for &p in &self.anchor_patterns[m.pattern().as_usize()] {
                    active[p] = true;
                }
            }
        }
        (0..active.len()).filter(|&i| active[i]).collect()
    }

    /// Validated candidates with byte offsets, before overlap resolution. Matching starts at byte
    /// `from`; the text before it is still visible to assertions such as `\b`.
    fn candidates(&self, text: &str, from: usize) -> Vec<Finding> {
        let mut out = Vec::new();
        for idx in self.active(&text[from..]) {
            let c = &self.patterns[idx];
            let mut push = |s: usize, e: usize, ms: usize, me: usize| {
                if s >= e {
                    return;
                }
                let factor = match &c.spec.validator {
                    Some(v) => validators::check(v, &text[s..e]),
                    None => Some(1.0),
                };
                if let Some(factor) = factor {
                    out.push(Finding {
                        start: s,
                        end: e,
                        match_start: ms,
                        match_end: me,
                        label: c.spec.label.clone(),
                        source: c.spec.name.clone(),
                        score: c.spec.score * factor,
                        rank: idx as u32,
                    });
                }
            };
            let mut pos = from;
            while pos <= text.len() {
                let (whole, span) = match c.group {
                    Some(g) => match c.regex.captures_at(text, pos) {
                        Some(caps) => {
                            let whole = caps.get(0).expect("group 0");
                            (
                                (whole.start(), whole.end()),
                                caps.get(g).map(|m| (m.start(), m.end())),
                            )
                        }
                        None => break,
                    },
                    None => match c.regex.find_at(text, pos) {
                        Some(m) => ((m.start(), m.end()), Some((m.start(), m.end()))),
                        None => break,
                    },
                };
                if let Some((s, e)) = span {
                    push(s, e, whole.0, whole.1);
                }
                pos = if whole.1 > whole.0 {
                    whole.1
                } else {
                    // Empty match: step over one character, as find_iter does.
                    match text[whole.1..].chars().next() {
                        Some(ch) => whole.1 + ch.len_utf8(),
                        None => break,
                    }
                };
            }
        }
        out
    }

    /// Unresolved candidates with character offsets.
    pub fn scan_candidates(&self, text: &str) -> Vec<Finding> {
        self.scan_candidates_from(text, 0)
    }

    /// Unresolved candidates whose matches start at or after character `from_char`.
    pub fn scan_candidates_from(&self, text: &str, from_char: usize) -> Vec<Finding> {
        let from = char_to_byte(text, from_char);
        let mut found = self.candidates(text, from);
        let mut offsets: Vec<usize> = found
            .iter()
            .flat_map(|f| [f.start, f.end, f.match_start, f.match_end])
            .collect();
        bytes_to_chars(text, &mut offsets);
        for (i, f) in found.iter_mut().enumerate() {
            f.start = offsets[4 * i];
            f.end = offsets[4 * i + 1];
            f.match_start = offsets[4 * i + 2];
            f.match_end = offsets[4 * i + 3];
        }
        found
    }

    /// Whether pattern `rank` is checked by a checksum or structure validator (cards, IBANs,
    /// IP addresses); its matches take precedence over overlapping tagger spans.
    pub fn is_structured(&self, rank: u32) -> bool {
        self.patterns
            .get(rank as usize)
            .and_then(|c| c.spec.validator.as_deref())
            .is_some_and(|v| validators::STRUCTURED.contains(&v))
    }

    /// Resolved, non-overlapping findings sorted by start (character offsets).
    pub fn scan(&self, text: &str) -> Vec<Finding> {
        resolve(self.scan_candidates(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, regex: &str, anchors: &[&str], validator: Option<&str>) -> PatternSpec {
        PatternSpec {
            name: name.into(),
            label: name.to_uppercase(),
            regex: regex.into(),
            group: None,
            anchors: anchors.iter().map(|s| s.to_string()).collect(),
            validator: validator.map(|s| s.to_string()),
            score: 0.9,
        }
    }

    #[test]
    fn prefilter_selects_patterns() {
        let engine = Engine::new(vec![
            spec("aws", r"\bAKIA[A-Z2-7]{16}\b", &["akia"], None),
            spec(
                "card",
                r"\b[0-9]{4}(?:[ -]?[0-9]{4}){3}\b",
                &[],
                Some("luhn"),
            ),
        ])
        .unwrap();
        assert_eq!(engine.active("nothing here"), vec![1]);
        assert_eq!(engine.active("key akia"), vec![0, 1]);
        let found =
            engine.scan("é AKIAABCDEFGHIJKLMNOP and 4111 1111 1111 1111 and 4111 1111 1111 1112");
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].start, found[0].end), (2, 22));
        assert_eq!(found[1].label, "CARD");
    }

    #[test]
    fn groups_and_errors() {
        let mut s = spec(
            "kv",
            r"(?i)password\s*[:=]\s*(?P<v>[^\s]{8,64})",
            &["password"],
            None,
        );
        s.group = Some("v".into());
        let engine = Engine::new(vec![s.clone()]).unwrap();
        let found = engine.scan("PASSWORD = hunter2hunter2");
        assert_eq!((found[0].start, found[0].end), (11, 25));
        s.group = Some("missing".into());
        assert!(Engine::new(vec![s]).is_err());
        assert!(Engine::new(vec![spec("bad", "(", &[], None)]).is_err());
        assert!(Engine::new(vec![spec("v", "a", &[], Some("nope"))]).is_err());
        assert!(Engine::from_json("{").is_err());
    }

    #[test]
    fn scanning_from_an_offset_keeps_context() {
        let engine = Engine::new(vec![spec("w", r"\bab\b", &[], None)]).unwrap();
        // "xab ab": scanning from char 1 must not see a word boundary before "ab".
        let found = engine.scan_candidates_from("xab ab", 1);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].start, 4);
        let all = engine.scan_candidates_from("é ab", 1);
        assert_eq!((all[0].start, all[0].end), (2, 4));
        assert!(engine.scan_candidates_from("ab", 9).is_empty());
    }
}
