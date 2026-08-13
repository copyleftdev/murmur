//! Finding the right weights on disk, and choosing between them.
//!
//! Parakeet ships in two precisions and the right one is not a matter of taste:
//! int8 is roughly a quarter of the size and matches fp32's speed *on a CPU*,
//! but on the CUDA provider most quantised operators have no kernel, so the
//! graph is copied back and forth across the bus and lands within noise of where
//! it started. fp32 is therefore the only sensible choice on a GPU, and int8 the
//! better one without, purely to save disk and memory.

use murmur_core::config::Precision;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Found {
    Fp32,
    Int8,
}

impl Found {
    #[must_use]
    pub fn precision(self) -> Precision {
        match self {
            Self::Fp32 => Precision::Fp32,
            Self::Int8 => Precision::Int8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variant {
    pub dir: PathBuf,
    pub kind: Found,
}

/// Does this directory hold a usable model, and at what precision?
///
/// fp32 wins when both are present in one directory, matching the order
/// `parakeet-rs` itself resolves encoder filenames in.
#[must_use]
pub fn classify(dir: &Path) -> Option<Found> {
    if !dir.join("vocab.txt").exists() {
        return None;
    }
    if dir.join("encoder-model.onnx").exists() {
        Some(Found::Fp32)
    } else if dir.join("encoder-model.int8.onnx").exists() {
        Some(Found::Int8)
    } else {
        None
    }
}

/// Every model under `root`, including `root` itself.
///
/// Accepting `root` as a model directory is what keeps an explicitly configured
/// path working after the default moved from one model to a directory of them.
#[must_use]
pub fn discover(root: &Path) -> Vec<Variant> {
    let mut found = Vec::new();
    if let Some(kind) = classify(root) {
        found.push(Variant { dir: root.to_path_buf(), kind });
    }
    if let Ok(entries) = std::fs::read_dir(root) {
        let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        dirs.sort();
        for dir in dirs {
            if let Some(kind) = classify(&dir) {
                found.push(Variant { dir, kind });
            }
        }
    }
    found
}

/// Pick the weights to load.
///
/// Pure, so the policy is decided by tests rather than by whatever happens to be
/// installed on the machine running them. An explicit `want` is honoured when
/// those weights exist and falls back rather than failing when they do not: a
/// slower model is better than no dictation.
#[must_use]
pub fn choose(variants: &[Variant], gpu_usable: bool, want: Precision) -> Option<&Variant> {
    let preferred = match want {
        Precision::Fp32 => Found::Fp32,
        Precision::Int8 => Found::Int8,
        Precision::Auto => {
            if gpu_usable {
                Found::Fp32
            } else {
                Found::Int8
            }
        }
    };
    variants
        .iter()
        .find(|v| v.kind == preferred)
        .or_else(|| variants.first())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp32(dir: &str) -> Variant {
        Variant { dir: PathBuf::from(dir), kind: Found::Fp32 }
    }

    fn int8(dir: &str) -> Variant {
        Variant { dir: PathBuf::from(dir), kind: Found::Int8 }
    }

    #[test]
    fn a_gpu_gets_fp32_because_int8_has_no_cuda_kernels() {
        let both = [int8("a"), fp32("b")];
        assert_eq!(choose(&both, true, Precision::Auto).unwrap().kind, Found::Fp32);
    }

    #[test]
    fn without_a_gpu_int8_is_chosen_to_save_disk_and_memory() {
        let both = [fp32("a"), int8("b")];
        assert_eq!(choose(&both, false, Precision::Auto).unwrap().kind, Found::Int8);
    }

    #[test]
    fn an_explicit_precision_overrides_the_hardware() {
        let both = [fp32("a"), int8("b")];
        assert_eq!(choose(&both, true, Precision::Int8).unwrap().kind, Found::Int8);
        assert_eq!(choose(&both, false, Precision::Fp32).unwrap().kind, Found::Fp32);
    }

    #[test]
    fn the_only_model_installed_is_used_whatever_was_preferred() {
        let only_int8 = [int8("a")];
        assert_eq!(choose(&only_int8, true, Precision::Auto).unwrap().kind, Found::Int8);
        assert_eq!(choose(&only_int8, true, Precision::Fp32).unwrap().kind, Found::Int8);

        let only_fp32 = [fp32("a")];
        assert_eq!(choose(&only_fp32, false, Precision::Auto).unwrap().kind, Found::Fp32);
        assert_eq!(choose(&only_fp32, false, Precision::Int8).unwrap().kind, Found::Fp32);
    }

    #[test]
    fn no_models_at_all_chooses_nothing_rather_than_guessing_a_path() {
        assert!(choose(&[], true, Precision::Auto).is_none());
    }

    #[test]
    fn choosing_is_stable_regardless_of_directory_order() {
        let one = [fp32("a"), int8("b")];
        let other = [int8("b"), fp32("a")];
        for gpu in [true, false] {
            assert_eq!(
                choose(&one, gpu, Precision::Auto).unwrap().kind,
                choose(&other, gpu, Precision::Auto).unwrap().kind,
                "gpu={gpu}"
            );
        }
    }

    #[test]
    fn a_directory_without_a_vocabulary_is_not_a_model() {
        let dir = std::env::temp_dir().join("murmur-not-a-model");
        std::fs::create_dir_all(&dir).expect("temp dir");
        assert_eq!(classify(&dir), None);
        assert!(discover(&dir).is_empty());
    }
}
