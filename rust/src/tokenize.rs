//! Tokenizer shared by the tagger: runs of alphanumeric characters (and `_`) form one token,
//! every other non-space character is a token of its own, whitespace separates.

/// Whitespace as Python's `str.isspace` sees it for the characters that matter here.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

pub fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Byte offsets `(start, end)` of each token.
pub fn tokenize(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut word_start: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if is_word(c) {
            if word_start.is_none() {
                word_start = Some(i);
            }
            continue;
        }
        if let Some(s) = word_start.take() {
            out.push((s, i));
        }
        if !is_space(c) {
            out.push((i, i + c.len_utf8()));
        }
    }
    if let Some(s) = word_start {
        out.push((s, text.len()));
    }
    out
}

/// Longest stretch of text, in characters, that the tagger decodes in one lattice.
pub const WINDOW_CHARS: usize = 16_384;

/// Split `text` into consecutive windows of at most `max_chars` characters, as
/// `(byte_start, byte_end, char_start)`. A window ends after the last newline in its second half,
/// else after the last space there, else at `max_chars`; newlines and spaces never belong to a
/// token, so only a run of `max_chars / 2` characters without either is ever cut inside a token.
/// Decoding windows separately bounds the forward-backward memory (tokens times tags) for long
/// documents; texts of at most `max_chars` characters form a single window.
pub fn windows(text: &str, max_chars: usize) -> Vec<(usize, usize, usize)> {
    assert!(max_chars >= 2, "windows need at least two characters");
    if text.len() <= max_chars {
        return vec![(0, text.len(), 0)];
    }
    let mut bytes: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
    let n = bytes.len();
    bytes.push(text.len());
    let mut out = Vec::new();
    let mut start = 0usize;
    while n - start > max_chars {
        let (lo, hi) = (start + max_chars / 2, start + max_chars);
        let region = &text[bytes[lo]..bytes[hi]];
        let cut = region
            .rfind('\n')
            .or_else(|| region.rfind(' '))
            .map_or(hi, |p| bytes.partition_point(|&b| b <= bytes[lo] + p));
        out.push((bytes[start], bytes[cut], start));
        start = cut;
    }
    out.push((bytes[start], text.len(), start));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_the_text_and_cut_at_whitespace() {
        assert_eq!(windows("short", 8), vec![(0, 5, 0)]);
        let text = "aé b\ncc dd ee ffffffffff";
        let w = windows(text, 8);
        let mut chars = 0;
        let mut byte = 0;
        for &(s, e, c) in &w {
            assert_eq!((s, c), (byte, chars));
            let part = &text[s..e];
            assert!(part.chars().count() <= 8);
            chars += part.chars().count();
            byte = e;
        }
        assert_eq!(byte, text.len());
        assert_eq!(&text[w[0].0..w[0].1], "aé b\n");
        assert_eq!(&text[w[1].0..w[1].1], "cc dd ");
        assert_eq!(&text[w[2].0..w[2].1], "ee fffff");
        assert_eq!(&text[w[3].0..], "fffff");
    }

    #[test]
    fn splits_words_and_punctuation() {
        let text = "Hi, Zoë! a_b 42.5\u{a0}x";
        let toks: Vec<&str> = tokenize(text).iter().map(|&(s, e)| &text[s..e]).collect();
        assert_eq!(
            toks,
            vec!["Hi", ",", "Zoë", "!", "a_b", "42", ".", "5", "x"]
        );
        assert!(tokenize("   ").is_empty());
    }
}
