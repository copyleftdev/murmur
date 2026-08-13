use anyhow::{Context, Result};
use murmur_core::Config;
use std::path::PathBuf;

/// Where the config lives, honouring `XDG_CONFIG_HOME`.
#[must_use]
pub fn path() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .unwrap_or_else(|| PathBuf::from(".config"))
        .join("murmur/config.toml")
}

/// Load the config, falling back to defaults when the file does not exist.
///
/// A missing config is the normal first-run state, not an error. A *malformed*
/// one is an error, and says which key it choked on rather than silently
/// reverting to defaults the user did not ask for.
///
/// # Errors
/// Fails if the file exists but cannot be read or parsed.
pub fn load() -> Result<Config> {
    let path = path();
    if !path.exists() {
        return Ok(Config::default());
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Write a fully commented default config, refusing to clobber an existing one.
///
/// # Errors
/// Fails if the file already exists or cannot be written.
pub fn write_default() -> Result<PathBuf> {
    let path = path();
    anyhow::ensure!(!path.exists(), "{} already exists", path.display());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let text = toml::to_string_pretty(&Config::default()).context("serialising defaults")?;
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Expand a leading `~` so config files can be written the way people type them.
#[must_use]
pub fn expand_home(input: &str) -> PathBuf {
    match input.strip_prefix("~/") {
        Some(rest) => std::env::var_os("HOME")
            .map_or_else(|| PathBuf::from(input), |home| PathBuf::from(home).join(rest)),
        None => PathBuf::from(input),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_path_ends_where_users_expect_it() {
        let path = path();
        assert!(path.ends_with("murmur/config.toml"), "{}", path.display());
    }

    #[test]
    fn home_is_expanded_only_at_the_start_of_a_path() {
        let home = std::env::var("HOME").unwrap();
        assert_eq!(expand_home("~/models/x"), PathBuf::from(&home).join("models/x"));
        assert_eq!(expand_home("/abs/~/x"), PathBuf::from("/abs/~/x"));
        assert_eq!(expand_home("relative/x"), PathBuf::from("relative/x"));
    }

    #[test]
    fn the_default_config_survives_a_toml_round_trip() {
        let text = toml::to_string_pretty(&Config::default()).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(back, Config::default());
    }
}
