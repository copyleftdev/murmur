use serde::{Deserialize, Serialize};

/// A user-defined replacement applied after transcription.
///
/// Parakeet cannot be biased toward a vocabulary at inference time, so proper
/// nouns and jargon are corrected here instead. `spoken` may be several words
/// (`"see sharp"` -> `"C#"`); it is matched against word boundaries only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DictEntry {
    pub spoken: String,
    pub written: String,
}

impl DictEntry {
    pub fn new(spoken: impl Into<String>, written: impl Into<String>) -> Self {
        Self { spoken: spoken.into(), written: written.into() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FormatConfig {
    /// Honour spoken editing commands such as "new paragraph" and "scratch that".
    pub commands: bool,
    /// Emit a trailing space so consecutive dictations flow into one another.
    pub trailing_space: bool,
    pub dictionary: Vec<DictEntry>,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self { commands: true, trailing_space: true, dictionary: Vec::new() }
    }
}

/// What the formatter needs to know about the text already at the cursor.
///
/// We cannot read the target application, so `continuation` is derived from our
/// own history: a second utterance within the continuation window is assumed to
/// be landing right after the first, and gets a separating space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmitContext {
    pub continuation: bool,
}

const BREAK_PHRASES: &[(&[&str], &str)] = &[
    (&["new", "paragraph"], "\n\n"),
    (&["new", "line"], "\n"),
    (&["next", "line"], "\n"),
];

const SCRATCH_PHRASES: &[&[&str]] = &[&["scratch", "that"], &["delete", "that"]];

#[derive(Clone, Debug)]
enum Piece {
    Word(String),
    Break(&'static str),
}

#[derive(Clone, Debug)]
struct Replacement {
    spoken: Vec<String>,
    written: String,
}

#[derive(Clone, Debug, Default)]
pub struct Formatter {
    config: FormatConfig,
    dictionary: Vec<Replacement>,
}

impl Formatter {
    #[must_use]
    pub fn new(config: FormatConfig) -> Self {
        let mut dictionary: Vec<Replacement> = config
            .dictionary
            .iter()
            .map(|entry| Replacement {
                spoken: entry.spoken.split_whitespace().map(str::to_ascii_lowercase).collect(),
                written: entry.written.clone(),
            })
            .filter(|r| !r.spoken.is_empty())
            .collect();
        // Longest phrase wins, so "see sharp" is not shadowed by an entry for "see".
        dictionary.sort_by_key(|r| std::cmp::Reverse(r.spoken.len()));
        Self { config, dictionary }
    }

    #[must_use]
    pub fn config(&self) -> &FormatConfig {
        &self.config
    }

    /// Turn a raw transcript into the exact bytes to inject at the cursor.
    ///
    /// Returns `None` when the utterance formats to nothing — an empty
    /// transcript, or one entirely consumed by a "scratch that".
    #[must_use]
    pub fn format(&self, raw: &str, ctx: EmitContext) -> Option<String> {
        let tokens: Vec<&str> = raw.split_whitespace().collect();
        let body = render(&self.rewrite(&tokens));
        if body.is_empty() {
            return None;
        }

        let mut out = String::with_capacity(body.len() + 2);
        if ctx.continuation && !body.starts_with('\n') {
            out.push(' ');
        }
        out.push_str(&body);
        if self.config.trailing_space && !body.ends_with('\n') {
            out.push(' ');
        }
        Some(out)
    }

    fn rewrite(&self, tokens: &[&str]) -> Vec<Piece> {
        let mut pieces: Vec<Piece> = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if self.config.commands {
                if let Some(len) =
                    SCRATCH_PHRASES.iter().find(|p| phrase_at(tokens, i, p)).map(|p| p.len())
                {
                    pieces.clear();
                    i += len;
                    continue;
                }
                if let Some((phrase, literal)) =
                    BREAK_PHRASES.iter().find(|(p, _)| phrase_at(tokens, i, p))
                {
                    pieces.push(Piece::Break(literal));
                    i += phrase.len();
                    continue;
                }
            }
            if let Some(entry) = self.dictionary.iter().find(|r| phrase_at(tokens, i, &r.spoken)) {
                let (_, tail) = split_trailing_punctuation(tokens[i + entry.spoken.len() - 1]);
                pieces.push(Piece::Word(format!("{}{tail}", entry.written)));
                i += entry.spoken.len();
                continue;
            }
            pieces.push(Piece::Word(tokens[i].to_owned()));
            i += 1;
        }
        pieces
    }
}

fn render(pieces: &[Piece]) -> String {
    let mut out = String::new();
    for piece in pieces {
        match piece {
            Piece::Break(literal) => {
                while out.ends_with(' ') {
                    out.pop();
                }
                // A leading break would indent the target application, not the text.
                if !out.is_empty() {
                    out.push_str(literal);
                }
            }
            Piece::Word(word) => {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push(' ');
                }
                out.push_str(word);
            }
        }
    }
    out.trim_end_matches(' ').to_owned()
}

/// Does `phrase` sit at `tokens[at..]`, ignoring case and ASR punctuation?
///
/// Parakeet punctuates its output, so the token for a spoken command is as
/// likely to be `"line."` as `"line"`.
fn phrase_at(tokens: &[&str], at: usize, phrase: &[impl AsRef<str>]) -> bool {
    !phrase.is_empty()
        && tokens.len() >= at + phrase.len()
        && phrase.iter().zip(&tokens[at..]).all(|(want, got)| {
            got.trim_matches(|c: char| c.is_ascii_punctuation())
                .eq_ignore_ascii_case(want.as_ref())
        })
}

fn split_trailing_punctuation(word: &str) -> (&str, &str) {
    word.split_at(word.trim_end_matches(|c: char| c.is_ascii_punctuation()).len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt() -> Formatter {
        Formatter::new(FormatConfig::default())
    }

    fn plain(raw: &str) -> Option<String> {
        fmt().format(raw, EmitContext::default())
    }

    fn with_dict(entries: Vec<DictEntry>) -> Formatter {
        Formatter::new(FormatConfig { dictionary: entries, ..FormatConfig::default() })
    }

    #[test]
    fn trailing_space_lets_dictations_flow() {
        assert_eq!(plain("Hello there.").as_deref(), Some("Hello there. "));
    }

    #[test]
    fn continuation_adds_a_leading_space() {
        let out = fmt().format("And another thing.", EmitContext { continuation: true });
        assert_eq!(out.as_deref(), Some(" And another thing. "));
    }

    #[test]
    fn break_commands_become_literal_newlines() {
        assert_eq!(plain("one new line two").as_deref(), Some("one\ntwo "));
        assert_eq!(plain("one new paragraph two").as_deref(), Some("one\n\ntwo "));
    }

    #[test]
    fn break_commands_survive_asr_punctuation() {
        assert_eq!(plain("One. New line. Two.").as_deref(), Some("One.\nTwo. "));
    }

    #[test]
    fn scratch_that_discards_everything_before_it() {
        let out = plain("the wrong words scratch that the right words");
        assert_eq!(out.as_deref(), Some("the right words "));
    }

    #[test]
    fn scratch_that_alone_emits_nothing() {
        assert_eq!(plain("some words scratch that"), None);
    }

    #[test]
    fn a_leading_break_does_not_indent_the_target() {
        assert_eq!(plain("new line hello").as_deref(), Some("hello "));
    }

    #[test]
    fn a_trailing_break_is_kept_and_suppresses_the_trailing_space() {
        assert_eq!(plain("hello new paragraph").as_deref(), Some("hello\n\n"));
    }

    #[test]
    fn dictionary_replaces_whole_words_and_keeps_punctuation() {
        let out = with_dict(vec![DictEntry::new("kubernetes", "Kubernetes")])
            .format("we deploy kubernetes, daily", EmitContext::default());
        assert_eq!(out.as_deref(), Some("we deploy Kubernetes, daily "));
    }

    #[test]
    fn dictionary_matches_multi_word_phrases() {
        let out = with_dict(vec![DictEntry::new("see sharp", "C#")])
            .format("written in see sharp.", EmitContext::default());
        assert_eq!(out.as_deref(), Some("written in C#. "));
    }

    #[test]
    fn longest_dictionary_phrase_wins() {
        let dict = vec![DictEntry::new("see", "C"), DictEntry::new("see sharp", "C#")];
        let out = with_dict(dict).format("see sharp", EmitContext::default());
        assert_eq!(out.as_deref(), Some("C# "));
    }

    #[test]
    fn dictionary_does_not_touch_substrings() {
        let out = with_dict(vec![DictEntry::new("cat", "CAT")])
            .format("the catalogue cat", EmitContext::default());
        assert_eq!(out.as_deref(), Some("the catalogue CAT "));
    }

    #[test]
    fn empty_and_whitespace_transcripts_emit_nothing() {
        assert_eq!(plain(""), None);
        assert_eq!(plain("   \n  "), None);
    }

    #[test]
    fn commands_can_be_disabled_and_pass_through_verbatim() {
        let f = Formatter::new(FormatConfig { commands: false, ..FormatConfig::default() });
        let out = f.format("one new line two", EmitContext::default());
        assert_eq!(out.as_deref(), Some("one new line two "));
    }

    #[test]
    fn formatting_never_emits_leading_or_doubled_spaces_in_the_body() {
        for raw in ["a  b", " leading", "trailing ", "a new line  b"] {
            let out = plain(raw).unwrap();
            assert!(!out.contains("  "), "{raw:?} produced {out:?}");
            assert!(!out.starts_with(' '), "{raw:?} produced {out:?}");
        }
    }
}
