//! Patterns and the tagger combined into one detector, with parallel batch scanning.

use std::sync::Arc;

use rayon::prelude::*;

use crate::crf::Crf;
use crate::engine::Engine;
use crate::finding::{char_to_byte, resolve, Finding};

pub struct Detector {
    pub engine: Engine,
    pub model: Option<Arc<Crf>>,
    /// Tagger spans with a lower mean marginal are dropped.
    pub ner_min_score: f64,
}

impl Detector {
    pub fn new(engine: Engine, model: Option<Arc<Crf>>, ner_min_score: f64) -> Detector {
        Detector {
            engine,
            model,
            ner_min_score,
        }
    }

    /// Pattern and tagger findings, resolved to non-overlapping spans sorted by start.
    pub fn scan(&self, text: &str) -> Vec<Finding> {
        resolve(self.candidates(text))
    }

    /// Pattern candidates and tagger spans above the score floor, before overlap resolution.
    pub fn candidates(&self, text: &str) -> Vec<Finding> {
        self.candidates_from(text, 0)
    }

    /// Candidates starting at or after character `from`; the tagger sees only `text[from..]`.
    /// Tagger spans that overlap a structured pattern match (see `Engine::is_structured`) are
    /// dropped, so a confident tagger span can neither relabel a Luhn-valid card nor cover part
    /// of an address and leave the rest visible.
    pub fn candidates_from(&self, text: &str, from: usize) -> Vec<Finding> {
        let mut found = self.engine.scan_candidates_from(text, from);
        if let Some(model) = &self.model {
            let mut structured: Vec<(usize, usize)> = found
                .iter()
                .filter(|f| self.engine.is_structured(f.rank))
                .map(|f| (f.start, f.end))
                .collect();
            structured.sort_unstable();
            // Running maximum of the ends: a span overlaps some structured match exactly when
            // the furthest end among the matches starting before its end lies past its start.
            let reach: Vec<usize> = structured
                .iter()
                .scan(0, |m, &(_, e)| {
                    *m = (*m).max(e);
                    Some(*m)
                })
                .collect();
            let overlaps = |s: usize, e: usize| {
                let k = structured.partition_point(|&(st, _)| st < e);
                k > 0 && reach[k - 1] > s
            };
            let byte = char_to_byte(text, from);
            let tagged: Vec<Finding> = model
                .predict(&text[byte..])
                .into_iter()
                .filter(|f| f.score >= self.ner_min_score)
                .map(|mut f| {
                    f.start += from;
                    f.end += from;
                    f.match_start += from;
                    f.match_end += from;
                    f
                })
                .filter(|f| !overlaps(f.start, f.end))
                .collect();
            found.extend(tagged);
        }
        found
    }

    /// `scan` over many texts on the rayon pool.
    pub fn scan_batch(&self, texts: &[String]) -> Vec<Vec<Finding>> {
        texts.par_iter().map(|t| self.scan(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::PatternSpec;

    #[test]
    fn batch_matches_single() {
        let engine = Engine::new(vec![PatternSpec {
            name: "email".into(),
            label: "EMAIL".into(),
            regex: r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}".into(),
            group: None,
            anchors: vec!["@".into()],
            validator: None,
            score: 0.9,
        }])
        .unwrap();
        let det = Detector::new(engine, None, 0.5);
        let texts: Vec<String> = (0..50)
            .map(|i| format!("mail u{i}@example.org now"))
            .collect();
        let batch = det.scan_batch(&texts);
        for (t, b) in texts.iter().zip(&batch) {
            assert_eq!(&det.scan(t), b);
            assert_eq!(b[0].start, 5);
        }
    }
}
