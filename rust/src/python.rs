//! PyO3 bindings, built by maturin into `llm_dlp_guard.llm_dlp_guard_rs`.

use std::sync::Arc;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use crate::crf::{Crf, TrainConfig, gold_tags as crf_gold_tags};
use crate::detector::Detector;
use crate::engine::Engine;
use crate::finding::{Finding, bytes_to_chars};
use crate::tokenize::tokenize as tokenize_bytes;
use crate::validators;

type Row = (usize, usize, String, String, f64, u32, usize, usize);

fn rows(found: Vec<Finding>) -> Vec<Row> {
    found
        .into_iter()
        .map(|f| {
            (
                f.start,
                f.end,
                f.label,
                f.source,
                f.score,
                f.rank,
                f.match_start,
                f.match_end,
            )
        })
        .collect()
}

fn with_threads<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> PyResult<T> {
    if threads == 0 {
        return Ok(work());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(pool.install(work))
}

/// The linear-chain CRF tagger.
#[pyclass(module = "llm_dlp_guard.llm_dlp_guard_rs", frozen)]
struct RustCrf {
    inner: Arc<Crf>,
}

#[pymethods]
impl RustCrf {
    #[staticmethod]
    fn load(py: Python<'_>, path: String) -> PyResult<Self> {
        let model = py
            .detach(|| Crf::load(&path))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(model),
        })
    }

    fn save(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.detach(|| self.inner.save(&path))
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner.labels.clone()
    }

    #[getter]
    fn n_buckets(&self) -> usize {
        self.inner.n_buckets
    }

    fn predict(&self, text: &str) -> Vec<Row> {
        rows(self.inner.predict(text))
    }

    /// Best tag path as tag ids (0 is O, then B and I per label).
    fn tag_path(&self, text: &str) -> Vec<usize> {
        self.inner.tag_path(text)
    }

    /// Feature buckets per token.
    fn feature_ids(&self, text: &str) -> Vec<Vec<u32>> {
        let toks = tokenize_bytes(text);
        self.inner
            .feature_ids(text, &toks)
            .chunks(crate::crf::FEATURES_PER_TOKEN)
            .map(|c| c.to_vec())
            .collect()
    }

    fn nll(&self, text: &str, gold: Vec<u8>) -> PyResult<f64> {
        let n = tokenize_bytes(text).len();
        let t = self.inner.num_tags();
        if gold.len() != n || gold.iter().any(|&g| g as usize >= t) {
            return Err(PyValueError::new_err("gold tags do not match the tokens"));
        }
        Ok(self.inner.nll(text, &gold))
    }
}

/// Patterns plus an optional tagger.
#[pyclass(module = "llm_dlp_guard.llm_dlp_guard_rs", frozen)]
struct RustDetector {
    inner: Detector,
}

#[pymethods]
impl RustDetector {
    #[new]
    #[pyo3(signature = (catalog_json, model=None, ner_min_score=0.5))]
    fn new(catalog_json: &str, model: Option<&RustCrf>, ner_min_score: f64) -> PyResult<Self> {
        let engine = Engine::from_json(catalog_json).map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Detector::new(engine, model.map(|m| m.inner.clone()), ner_min_score),
        })
    }

    fn scan(&self, py: Python<'_>, text: &str) -> Vec<Row> {
        py.detach(|| rows(self.inner.scan(text)))
    }

    /// Pattern and tagger candidates before overlap resolution, matching from character `start`.
    #[pyo3(signature = (text, start=0))]
    fn candidates(&self, py: Python<'_>, text: &str, start: usize) -> Vec<Row> {
        py.detach(|| rows(self.inner.candidates_from(text, start)))
    }

    /// Catalog indices selected by the Aho-Corasick prefilter.
    fn active(&self, text: &str) -> Vec<usize> {
        self.inner.engine.active(text)
    }

    #[pyo3(signature = (texts, threads=0))]
    fn scan_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        threads: usize,
    ) -> PyResult<Vec<Vec<Row>>> {
        let out = py.detach(|| with_threads(threads, || self.inner.scan_batch(&texts)))?;
        Ok(out.into_iter().map(rows).collect())
    }
}

/// Train the tagger on character-offset spans.
#[pyfunction]
#[pyo3(signature = (texts, spans, labels, n_buckets=1<<20, epochs=3, learning_rate=0.1, batch_size=64, seed=0, verbose=false, threads=0))]
#[allow(clippy::too_many_arguments)]
fn train_crf(
    py: Python<'_>,
    texts: Vec<String>,
    spans: Vec<Vec<(usize, usize, String)>>,
    labels: Vec<String>,
    n_buckets: usize,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    seed: u64,
    verbose: bool,
    threads: usize,
) -> PyResult<RustCrf> {
    if texts.len() != spans.len() {
        return Err(PyValueError::new_err("texts and spans differ in length"));
    }
    if n_buckets == 0 || labels.is_empty() || labels.len() > 120 {
        return Err(PyValueError::new_err(
            "need n_buckets > 0 and 1 to 120 labels",
        ));
    }
    let cfg = TrainConfig {
        n_buckets,
        epochs,
        learning_rate,
        batch_size,
        seed,
        verbose,
    };
    let model = py.detach(|| with_threads(threads, || Crf::train(&texts, &spans, labels, &cfg)))?;
    Ok(RustCrf {
        inner: Arc::new(model),
    })
}

/// Token character offsets.
#[pyfunction]
fn tokenize(text: &str) -> Vec<(usize, usize)> {
    let toks = tokenize_bytes(text);
    let mut offsets: Vec<usize> = toks.iter().flat_map(|&(s, e)| [s, e]).collect();
    bytes_to_chars(text, &mut offsets);
    offsets.chunks(2).map(|c| (c[0], c[1])).collect()
}

/// BIO tag ids for character-offset spans.
#[pyfunction]
fn gold_tags(text: &str, spans: Vec<(usize, usize, String)>, labels: Vec<String>) -> Vec<u8> {
    crf_gold_tags(text, &tokenize_bytes(text), &spans, &labels)
}

/// Run one validator; `None` means rejected.
#[pyfunction]
fn validate(name: &str, value: &str) -> Option<f64> {
    validators::check(name, value)
}

#[pyfunction]
fn shannon_entropy(value: &str) -> f64 {
    validators::shannon_entropy(value)
}

#[pymodule]
fn llm_dlp_guard_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustCrf>()?;
    m.add_class::<RustDetector>()?;
    m.add_function(wrap_pyfunction!(train_crf, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(gold_tags, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(shannon_entropy, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
