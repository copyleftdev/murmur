use evdev::KeyCode;

/// Which key, and whether Shift is held, to produce a character on a US layout.
///
/// A `/dev/uinput` device emits scancodes, not characters: the compositor applies
/// the keymap. We therefore cannot type a character whose keysym the *user's*
/// layout does not have, and we cannot know their layout from here. Anything this
/// table cannot express is routed to the clipboard backend instead, which is
/// layout- and Unicode-independent.
#[must_use]
pub fn key_for(c: char) -> Option<(KeyCode, bool)> {
    use KeyCode as K;
    let shifted = |k: KeyCode| Some((k, true));
    let plain = |k: KeyCode| Some((k, false));
    match c {
        'a'..='z' => plain(letter(c)),
        'A'..='Z' => shifted(letter(c.to_ascii_lowercase())),
        '1' => plain(K::KEY_1),
        '2' => plain(K::KEY_2),
        '3' => plain(K::KEY_3),
        '4' => plain(K::KEY_4),
        '5' => plain(K::KEY_5),
        '6' => plain(K::KEY_6),
        '7' => plain(K::KEY_7),
        '8' => plain(K::KEY_8),
        '9' => plain(K::KEY_9),
        '0' => plain(K::KEY_0),
        '!' => shifted(K::KEY_1),
        '@' => shifted(K::KEY_2),
        '#' => shifted(K::KEY_3),
        '$' => shifted(K::KEY_4),
        '%' => shifted(K::KEY_5),
        '^' => shifted(K::KEY_6),
        '&' => shifted(K::KEY_7),
        '*' => shifted(K::KEY_8),
        '(' => shifted(K::KEY_9),
        ')' => shifted(K::KEY_0),
        ' ' => plain(K::KEY_SPACE),
        '\n' => plain(K::KEY_ENTER),
        '\t' => plain(K::KEY_TAB),
        '-' => plain(K::KEY_MINUS),
        '_' => shifted(K::KEY_MINUS),
        '=' => plain(K::KEY_EQUAL),
        '+' => shifted(K::KEY_EQUAL),
        '[' => plain(K::KEY_LEFTBRACE),
        '{' => shifted(K::KEY_LEFTBRACE),
        ']' => plain(K::KEY_RIGHTBRACE),
        '}' => shifted(K::KEY_RIGHTBRACE),
        '\\' => plain(K::KEY_BACKSLASH),
        '|' => shifted(K::KEY_BACKSLASH),
        ';' => plain(K::KEY_SEMICOLON),
        ':' => shifted(K::KEY_SEMICOLON),
        '\'' => plain(K::KEY_APOSTROPHE),
        '"' => shifted(K::KEY_APOSTROPHE),
        '`' => plain(K::KEY_GRAVE),
        '~' => shifted(K::KEY_GRAVE),
        ',' => plain(K::KEY_COMMA),
        '<' => shifted(K::KEY_COMMA),
        '.' => plain(K::KEY_DOT),
        '>' => shifted(K::KEY_DOT),
        '/' => plain(K::KEY_SLASH),
        '?' => shifted(K::KEY_SLASH),
        _ => None,
    }
}

fn letter(lower: char) -> KeyCode {
    use KeyCode as K;
    const LETTERS: [KeyCode; 26] = [
        K::KEY_A,
        K::KEY_B,
        K::KEY_C,
        K::KEY_D,
        K::KEY_E,
        K::KEY_F,
        K::KEY_G,
        K::KEY_H,
        K::KEY_I,
        K::KEY_J,
        K::KEY_K,
        K::KEY_L,
        K::KEY_M,
        K::KEY_N,
        K::KEY_O,
        K::KEY_P,
        K::KEY_Q,
        K::KEY_R,
        K::KEY_S,
        K::KEY_T,
        K::KEY_U,
        K::KEY_V,
        K::KEY_W,
        K::KEY_X,
        K::KEY_Y,
        K::KEY_Z,
    ];
    LETTERS[(lower as u8 - b'a') as usize]
}

/// The character a key and shift state produce, inverting [`key_for`].
///
/// Used to decode our own emitted scancodes during `murmur selftest`, which is
/// how the injection path is verified without a focused window to type into.
#[must_use]
pub fn char_for(key: KeyCode, shift: bool) -> Option<char> {
    ('\t'..='\u{7e}').find(|c| key_for(*c) == Some((key, shift)))
}

/// Can every character in `text` be typed as a scancode?
#[must_use]
pub fn is_typable(text: &str) -> bool {
    text.chars().all(|c| key_for(c).is_some())
}

/// The first character that cannot be typed, for diagnostics.
#[must_use]
pub fn first_untypable(text: &str) -> Option<char> {
    text.chars().find(|c| key_for(*c).is_none())
}

/// Every key the virtual device must declare, including modifiers.
#[must_use]
pub fn all_keys() -> Vec<KeyCode> {
    let mut keys = vec![
        KeyCode::KEY_LEFTSHIFT,
        KeyCode::KEY_LEFTCTRL,
        KeyCode::KEY_LEFTALT,
        KeyCode::KEY_BACKSPACE,
    ];
    keys.extend((0x20u8..0x7f).filter_map(|b| key_for(b as char).map(|(k, _)| k)));
    keys.push(KeyCode::KEY_ENTER);
    keys.push(KeyCode::KEY_TAB);
    keys.sort_unstable_by_key(|k| k.code());
    keys.dedup();
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_printable_ascii_character_is_typable() {
        for b in 0x20u8..0x7f {
            let c = b as char;
            assert!(key_for(c).is_some(), "no key for {c:?}");
        }
        assert!(key_for('\n').is_some());
        assert!(key_for('\t').is_some());
    }

    #[test]
    fn distinct_characters_never_share_a_key_and_shift_state() {
        let mut seen = std::collections::HashMap::new();
        for b in 0x20u8..0x7f {
            let c = b as char;
            let combo = key_for(c).unwrap();
            if let Some(other) = seen.insert((combo.0.code(), combo.1), c) {
                panic!("{c:?} and {other:?} both map to {combo:?}");
            }
        }
    }

    #[test]
    fn case_differs_only_by_shift() {
        for lower in 'a'..='z' {
            let upper = lower.to_ascii_uppercase();
            let (lk, lshift) = key_for(lower).unwrap();
            let (uk, ushift) = key_for(upper).unwrap();
            assert_eq!(lk.code(), uk.code());
            assert!(!lshift && ushift);
        }
    }

    #[test]
    fn every_typable_character_survives_a_scancode_round_trip() {
        for b in 0x20u8..0x7f {
            let c = b as char;
            let (key, shift) = key_for(c).unwrap();
            assert_eq!(char_for(key, shift), Some(c), "round trip failed for {c:?}");
        }
        for c in ['\n', '\t'] {
            let (key, shift) = key_for(c).unwrap();
            assert_eq!(char_for(key, shift), Some(c));
        }
    }

    #[test]
    fn non_ascii_is_rejected_so_it_can_be_routed_to_the_clipboard() {
        assert!(!is_typable("naïve"));
        assert_eq!(first_untypable("caf\u{e9} time"), Some('\u{e9}'));
        assert_eq!(first_untypable("plain ascii"), None);
    }

    #[test]
    fn the_declared_key_set_covers_everything_the_table_can_emit() {
        let declared: std::collections::HashSet<u16> =
            all_keys().iter().map(|k| k.code()).collect();
        for b in 0x20u8..0x7f {
            let (key, _) = key_for(b as char).unwrap();
            assert!(
                declared.contains(&key.code()),
                "{:?} not declared",
                b as char
            );
        }
        assert!(declared.contains(&KeyCode::KEY_LEFTCTRL.code()));
        assert!(declared.contains(&KeyCode::KEY_LEFTSHIFT.code()));
    }
}
