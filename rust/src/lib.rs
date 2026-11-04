//! Rust core of llm-dlp-guard.
//!
//! * [`engine`]: the pattern engine (an Aho-Corasick prefilter over literal anchors, then the
//!   regular expressions it selects, then validators and overlap resolution).
//! * [`validators`]: Luhn, IBAN mod-97, IP ranges, JWT structure, GitHub token checksums,
//!   US SSN rules and Shannon entropy.
//! * [`tokenize`] and [`crf`]: the tokenizer and the linear-chain CRF tagger for names,
//!   addresses and other free-form PII, with training.
//! * [`detector`]: patterns and the tagger combined, with batch scanning on all cores.
//!
//! All offsets returned by public functions are character (Unicode code point) offsets, the
//! same indices Python uses for `str`.

pub mod crf;
pub mod detector;
pub mod engine;
pub mod finding;
pub mod tokenize;
pub mod validators;

#[cfg(feature = "python")]
mod python;
