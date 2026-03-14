//! Rust-accelerated work-stealing deque for PyAccelerate.
//!
//! Uses `crossbeam-deque` (the same algorithm Tokio uses) wrapped
//! in a PyO3 extension so Python code can call it with minimal overhead.
//!
//! Build with maturin:
//!     cd bindings/rust/pyaccelerate_native
//!     maturin develop --release

use crossbeam_deque::{Injector, Steal, Worker as CbWorker};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A single work-stealing local deque backed by `crossbeam_deque::Worker`.
///
/// The owner thread pushes/pops from the back (LIFO).
/// Other threads steal from the front (FIFO) through the `Stealer`.
#[pyclass]
struct FastDeque {
    worker: CbWorker<PyObject>,
    push_count: AtomicU64,
    steal_count: Arc<AtomicU64>,
}

#[pymethods]
impl FastDeque {
    #[new]
    fn new() -> Self {
        Self {
            worker: CbWorker::new_fifo(),
            push_count: AtomicU64::new(0),
            steal_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Push to bottom (owner side, lock-free).
    fn push(&self, py: Python<'_>, item: PyObject) {
        self.worker.push(item);
        self.push_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop from bottom (LIFO, owner side, lock-free).
    fn pop(&self, py: Python<'_>) -> Option<PyObject> {
        self.worker.pop()
    }

    /// Steal from top (FIFO, any thread). Returns None if empty.
    fn steal(&self, py: Python<'_>) -> Option<PyObject> {
        let stealer = self.worker.stealer();
        loop {
            match stealer.steal() {
                Steal::Success(item) => {
                    self.steal_count.fetch_add(1, Ordering::Relaxed);
                    return Some(item);
                }
                Steal::Empty => return None,
                Steal::Retry => continue,
            }
        }
    }

    /// Number of items (approximate).
    fn __len__(&self) -> usize {
        // crossbeam Worker doesn't expose len; approximate via counters
        let pushed = self.push_count.load(Ordering::Relaxed);
        let stolen = self.steal_count.load(Ordering::Relaxed);
        (pushed.saturating_sub(stolen)) as usize
    }

    #[getter]
    fn is_empty(&self) -> bool {
        self.worker.is_empty()
    }

    #[getter]
    fn get_push_count(&self) -> u64 {
        self.push_count.load(Ordering::Relaxed)
    }

    #[getter]
    fn get_steal_count(&self) -> u64 {
        self.steal_count.load(Ordering::Relaxed)
    }
}

/// Python module definition.
#[pymodule]
fn pyaccelerate_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastDeque>()?;
    Ok(())
}
