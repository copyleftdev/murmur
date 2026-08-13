//! Making ONNX Runtime's CUDA provider loadable without touching the system.
//!
//! The provider ships linked against a specific CUDA major version — 13, at the
//! time of writing — and distributions lag well behind that. The usual advice is
//! to install a system CUDA toolkit, which needs root, conflicts with whatever
//! the distribution already ships, and is awkward to undo.
//!
//! None of that is necessary. The libraries are ordinary userspace shared
//! objects; only the *driver* is privileged, and the driver is already installed.
//! So Murmur keeps its own copy in its data directory and loads it directly.
//!
//! `LD_LIBRARY_PATH` cannot help here, because glibc reads it once at process
//! start — setting it from inside the process is too late. Instead each library
//! is opened with `RTLD_GLOBAL`, which puts it in the global symbol namespace,
//! and the loader then satisfies every later `DT_NEEDED` and `dlopen`-by-soname
//! from what is already resident. That is also how cuDNN 9 finds its own
//! sub-libraries, which it opens by bare name at runtime.

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_LAZY};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, Layer};

/// How many times ONNX Runtime has failed to register an execution provider.
static REGISTRATION_FAILURES: AtomicUsize = AtomicUsize::new(0);

/// Observes ONNX Runtime's own verdict on execution-provider registration.
///
/// ORT does not return an error when a provider fails to register — it logs one
/// and carries on using the CPU. Since `ort` bridges that log into `tracing`,
/// watching for it is not a heuristic: it is ORT telling us directly. Install
/// this layer alongside your subscriber and the model can report the device it
/// actually ran on instead of the one it asked for.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProviderWatch;

impl<S: tracing::Subscriber> Layer<S> for ProviderWatch {
    fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
        if *event.metadata().level() != tracing::Level::ERROR {
            return;
        }
        let mut looking = FailedRegistration(false);
        event.record(&mut looking);
        if looking.0 {
            REGISTRATION_FAILURES.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct FailedRegistration(bool);

impl Visit for FailedRegistration {
    fn record_debug(&mut self, _: &Field, value: &dyn std::fmt::Debug) {
        let text = format!("{value:?}");
        self.0 |= text.contains("attempting to register") && text.contains("ExecutionProvider");
    }
}

/// A monotonically increasing count of provider registration failures.
#[must_use]
pub fn registration_failures() -> usize {
    REGISTRATION_FAILURES.load(Ordering::Relaxed)
}


const PROVIDER: &str = "libonnxruntime_providers_cuda.so";

/// Where Murmur keeps a private CUDA runtime, if the system has none.
#[must_use]
pub fn bundled_dir() -> PathBuf {
    if let Some(explicit) = std::env::var_os("MURMUR_CUDA_DIR") {
        return PathBuf::from(explicit);
    }
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("murmur/cuda/lib")
}

/// Load every shared object in `dir` into the global symbol namespace.
///
/// Dependency order is discovered rather than hardcoded: each pass loads
/// whatever will load, and passes repeat while progress is being made. A
/// library whose dependency has not been loaded yet simply fails this pass and
/// succeeds on a later one, which keeps this correct as CUDA's internal library
/// graph changes between releases.
///
/// Handles are deliberately leaked. Unloading a CUDA library out from under
/// ONNX Runtime would be far worse than the memory it holds.
pub fn preload(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut pending: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains(".so") && !n.contains("stub"))
        })
        .collect();
    pending.sort();

    let mut loaded = 0usize;
    loop {
        let before = pending.len();
        pending.retain(|path| {
            // SAFETY: these are NVIDIA's own runtime libraries, and ONNX Runtime
            // is about to load the same files itself.
            match unsafe { Library::open(Some(path), RTLD_LAZY | RTLD_GLOBAL) } {
                Ok(library) => {
                    std::mem::forget(library);
                    loaded += 1;
                    false
                }
                Err(_) => true,
            }
        });
        if pending.len() == before {
            return loaded;
        }
    }
}

/// Make the CUDA runtime resident before ONNX Runtime asks for it.
///
/// Returns how many libraries were loaded from Murmur's own directory; zero
/// simply means the system is expected to provide them.
///
/// Deliberately *not* a test of the provider itself. Loading
/// `libonnxruntime_providers_cuda.so` outside ONNX Runtime's own initialisation
/// is not a safe probe: it imports `Provider_GetHost` from the provider bridge
/// and runs static initialisers that expect a CUDA context, so it fails — or
/// crashes — for reasons that never arise in the real load path.
pub fn ensure_runtime() -> usize {
    let dir = bundled_dir();
    if !dir.is_dir() {
        return 0;
    }
    let count = preload(&dir);
    tracing::info!(libraries = count, dir = %dir.display(), "preloaded bundled CUDA runtime");
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundled_directory_lives_under_the_data_home() {
        let dir = bundled_dir();
        assert!(dir.ends_with("murmur/cuda/lib"), "{}", dir.display());
    }

    #[test]
    fn an_explicit_directory_overrides_the_default() {
        // SAFETY: single-threaded test, restored immediately.
        unsafe { std::env::set_var("MURMUR_CUDA_DIR", "/tmp/murmur-cuda-test") };
        assert_eq!(bundled_dir(), PathBuf::from("/tmp/murmur-cuda-test"));
        unsafe { std::env::remove_var("MURMUR_CUDA_DIR") };
    }

    #[test]
    fn preloading_a_directory_that_does_not_exist_is_not_an_error() {
        assert_eq!(preload(Path::new("/nonexistent/murmur/cuda")), 0);
    }

    #[test]
    fn preloading_a_directory_with_no_libraries_loads_nothing() {
        let dir = std::env::temp_dir().join("murmur-empty-cuda");
        std::fs::create_dir_all(&dir).expect("temp dir");
        assert_eq!(preload(&dir), 0);
    }
}
