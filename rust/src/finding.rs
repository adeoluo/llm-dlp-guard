//! Findings, offset conversion and overlap resolution shared by every detector.

use std::cmp::Ordering;

/// Rank given to tagger findings; pattern findings use their catalog index.
pub const NER_RANK: u32 = 1_000_000;

/// One detected span. `start` and `end` are character offsets, `end` exclusive.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub start: usize,
    pub end: usize,
    /// Extent of the whole regular-expression match (equal to the span unless the pattern
    /// reports a capture group); streaming never cuts inside it.
    pub match_start: usize,
    pub match_end: usize,
    pub label: String,
    pub source: String,
    pub score: f64,
    pub rank: u32,
}

/// Map byte offsets of `text` to character offsets in place.
pub fn bytes_to_chars(text: &str, offsets: &mut [usize]) {
    if text.is_ascii() || offsets.is_empty() {
        return;
    }
    let mut map = vec![0usize; text.len() + 1];
    let mut chars = 0usize;
    for (byte, _) in text.char_indices() {
        map[byte] = chars;
        chars += 1;
    }
    map[text.len()] = chars;
    for offset in offsets.iter_mut() {
        *offset = map[*offset];
    }
}

/// Byte offset of character `index` (clamped to the end of `text`).
pub fn char_to_byte(text: &str, index: usize) -> usize {
    if text.is_ascii() {
        return index.min(text.len());
    }
    text.char_indices()
        .nth(index)
        .map_or(text.len(), |(b, _)| b)
}

fn priority(a: &Finding, b: &Finding) -> Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| (b.end - b.start).cmp(&(a.end - a.start)))
        .then_with(|| a.start.cmp(&b.start))
        .then_with(|| a.rank.cmp(&b.rank))
}

/// Keep a non-overlapping subset: highest score first, then the longer span, then the earlier
/// start, then the lower rank (catalog order). The result is sorted by start.
pub fn resolve(mut candidates: Vec<Finding>) -> Vec<Finding> {
    if candidates.len() <= 1 {
        return candidates;
    }
    candidates.sort_by(priority);
    // `kept` stays sorted by start; spans in it never overlap.
    let mut kept: Vec<Finding> = Vec::with_capacity(candidates.len());
    for cand in candidates {
        let idx = kept.partition_point(|f| f.start < cand.start);
        let clash_prev = idx > 0 && kept[idx - 1].end > cand.start;
        let clash_next = idx < kept.len() && kept[idx].start < cand.end;
        if !clash_prev && !clash_next {
            kept.insert(idx, cand);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(start: usize, end: usize, score: f64, rank: u32) -> Finding {
        Finding {
            start,
            end,
            match_start: start,
            match_end: end,
            label: "X".into(),
            source: "t".into(),
            score,
            rank,
        }
    }

    #[test]
    fn resolve_prefers_score_then_length() {
        let out = resolve(vec![f(0, 5, 0.5, 0), f(3, 10, 0.9, 1), f(10, 12, 0.1, 2)]);
        assert_eq!(out, vec![f(3, 10, 0.9, 1), f(10, 12, 0.1, 2)]);
        let out = resolve(vec![f(0, 4, 0.5, 0), f(0, 8, 0.5, 1)]);
        assert_eq!(out, vec![f(0, 8, 0.5, 1)]);
    }

    #[test]
    fn char_offsets() {
        let text = "é a";
        let mut offs = vec![0, 2, 3, 4];
        bytes_to_chars(text, &mut offs);
        assert_eq!(offs, vec![0, 1, 2, 3]);
    }
}
