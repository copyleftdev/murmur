//! Fetching weights, so an installed Murmur can be made to work.
//!
//! A package cannot carry the model: int8 is 640 MB and full precision 2.5 GB.
//! But telling a user who just installed something to assemble a `curl` loop is
//! not an install experience, so this does it — resumably, and without leaving a
//! half-written model behind to be discovered later as a corrupt one.

use anyhow::{Context, Result, bail};
use murmur_core::config::Precision;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const REPO: &str = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main";

/// Files shared by both precisions.
const COMMON: &[&str] = &["vocab.txt", "config.json", "nemo128.onnx"];

const INT8: &[&str] = &["decoder_joint-model.int8.onnx", "encoder-model.int8.onnx"];

const FP32: &[&str] = &[
    "decoder_joint-model.onnx",
    "encoder-model.onnx",
    "encoder-model.onnx.data",
];

/// The directory name a precision is installed under.
///
/// Separate directories rather than one, because the selector chooses between
/// whole models and a directory holding both precisions would be ambiguous.
#[must_use]
pub fn directory_for(precision: Precision) -> &'static str {
    match precision {
        Precision::Fp32 => "parakeet-tdt-0.6b-v3-fp32",
        _ => "parakeet-tdt-0.6b-v3",
    }
}

/// Download the weights for `precision` into `root`.
///
/// # Errors
/// Fails if the download cannot be completed or written.
pub fn pull(root: &Path, precision: Precision, gpu_usable: bool) -> Result<PathBuf> {
    // `Auto` here means the same thing it means at load time: full precision is
    // only worth its size if there is a GPU to run it on.
    let precision = match precision {
        Precision::Auto if gpu_usable => Precision::Fp32,
        Precision::Auto => Precision::Int8,
        explicit => explicit,
    };

    let dir = root.join(directory_for(precision));
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let files: Vec<&str> = COMMON
        .iter()
        .chain(if precision == Precision::Fp32 {
            FP32
        } else {
            INT8
        })
        .copied()
        .collect();

    println!("  into    {}", dir.display());
    println!("  weights {precision:?}\n");

    for name in files {
        fetch(&dir.join(name), &format!("{REPO}/{name}"), name)?;
    }

    println!("\n  done. `murmur doctor` will now show this model.");
    Ok(dir)
}

/// Fetch one file, skipping it if it is already complete.
fn fetch(destination: &Path, url: &str, name: &str) -> Result<()> {
    // Identity encoding on purpose: Hugging Face gzips the small text files, and
    // a compressed content-length cannot be compared with the size on disk --
    // which silently re-downloaded every one of them on each run.
    let mut response = ureq::get(url)
        .header("accept-encoding", "identity")
        .call()
        .with_context(|| format!("requesting {name}"))?;
    let expected = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());

    if let (Ok(existing), Some(expected)) = (std::fs::metadata(destination), expected)
        && existing.len() == expected
    {
        println!("  \u{2713} {name} (already present)");
        return Ok(());
    }

    // Written beside the target and renamed at the end: an interrupted download
    // must not leave something that looks like a usable model.
    let partial = destination.with_extension("part");
    let mut file = std::fs::File::create(&partial)
        .with_context(|| format!("creating {}", partial.display()))?;

    // Only the large files earn a progress bar; drawing one for a 90 KB
    // vocabulary just produces a line of noise per file.
    let show_progress = expected.is_none_or(|size| size > 4 << 20);

    let mut reader = response.body_mut().as_reader();
    let mut buffer = vec![0u8; 1 << 20];
    let mut written = 0u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("reading {name}"))?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .with_context(|| format!("writing {name}"))?;
        written += read as u64;
        if show_progress {
            progress(name, written, expected);
        }
    }
    file.sync_all().ok();
    drop(file);

    if let Some(expected) = expected
        && written != expected
    {
        let _ = std::fs::remove_file(&partial);
        bail!("{name} stopped after {written} of {expected} bytes");
    }

    std::fs::rename(&partial, destination)
        .with_context(|| format!("finishing {}", destination.display()))?;
    if show_progress {
        eprintln!();
    } else {
        println!("  \u{2713} {name}");
    }
    Ok(())
}

fn progress(name: &str, written: u64, expected: Option<u64>) {
    const STEP: u64 = 8 << 20;
    if written % STEP > (1 << 20) {
        return;
    }
    let megabytes = written as f64 / 1e6;
    match expected {
        Some(total) if total > 0 => {
            #[allow(clippy::cast_precision_loss)]
            let share = written as f64 / total as f64;
            let cells = (share * 24.0).round() as usize;
            eprint!(
                "\r  \u{2193} {name:<34} {}{} {megabytes:6.0} MB",
                "\u{2588}".repeat(cells),
                "\u{2591}".repeat(24 - cells.min(24)),
            );
        }
        _ => eprint!("\r  \u{2193} {name:<34} {megabytes:6.0} MB"),
    }
    let _ = std::io::stderr().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_precision_installs_somewhere_of_its_own() {
        assert_ne!(
            directory_for(Precision::Fp32),
            directory_for(Precision::Int8)
        );
    }

    #[test]
    fn auto_installs_where_auto_would_look() {
        // The selector treats a directory without full-precision weights as the
        // int8 model, so `auto` must not put int8 in the fp32 directory.
        assert_eq!(
            directory_for(Precision::Auto),
            directory_for(Precision::Int8)
        );
    }

    #[test]
    fn the_two_file_lists_do_not_overlap() {
        for file in INT8 {
            assert!(!FP32.contains(file), "{file} is claimed by both precisions");
        }
    }

    #[test]
    fn every_file_list_is_complete_enough_to_load() {
        // `murmur-asr` recognises a Parakeet model by its vocabulary and encoder.
        assert!(COMMON.contains(&"vocab.txt"));
        assert!(INT8.iter().any(|f| f.starts_with("encoder-model")));
        assert!(FP32.iter().any(|f| f.starts_with("encoder-model")));
        assert!(
            FP32.contains(&"encoder-model.onnx.data"),
            "external weights are required"
        );
    }
}
