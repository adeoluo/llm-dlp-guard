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

#[cfg(test)]
mod tests {
    use super::*;

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
