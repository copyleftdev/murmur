use crate::{InjectError, Result};
use std::io::Read;
use wl_clipboard_rs::copy::{MimeType, Options, ServeRequests, Source};
use wl_clipboard_rs::paste::{self, ClipboardType, Seat};

/// Read the clipboard as UTF-8 text, if it currently holds any.
///
/// A missing or non-text clipboard is not an error — it is the common case on a
/// freshly booted session — so both are reported as `None`.
#[must_use]
pub fn snapshot() -> Option<String> {
    let (mut reader, _) = paste::get_contents(
        ClipboardType::Regular,
        Seat::Unspecified,
        paste::MimeType::Text,
    )
    .ok()?;
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

/// Claim the clipboard and serve `text` to whoever asks for it.
///
/// `serve_requests` bounds how many times the data is handed out before this
/// thread lets go. One is right for a paste we are about to trigger ourselves;
/// unlimited is right when handing the user's own clipboard back afterwards.
fn offer(text: &str, serve_requests: ServeRequests) -> Result<()> {
    let mut options = Options::new();
    options
        .serve_requests(serve_requests)
        // Our text is exact: a trailing space is a deliberate part of the emission.
        .trim_newline(false)
        .foreground(true);

    let prepared = options
        .prepare_copy(Source::Bytes(text.as_bytes().into()), MimeType::Text)
        .map_err(|e| InjectError::Clipboard(e.to_string()))?;

    std::thread::spawn(move || {
        if let Err(error) = prepared.serve() {
            tracing::debug!(%error, "clipboard offer ended");
        }
    });
    Ok(())
}

/// Put `text` on the clipboard for exactly one consumer.
///
/// # Errors
/// Fails if the compositor exposes no data-control protocol.
pub fn offer_once(text: &str) -> Result<()> {
    offer(text, ServeRequests::Only(1))
}

/// Hand a previously captured clipboard back to the user.
///
/// # Errors
/// Fails if the compositor exposes no data-control protocol.
pub fn restore(text: &str) -> Result<()> {
    offer(text, ServeRequests::Unlimited)
}

/// Is a clipboard reachable from a background process on this session?
///
/// Setting the clipboard without holding keyboard focus needs a data-control
/// protocol (`wlr-data-control` or `ext-data-control`). Compositors that expose
/// neither will fail here, and the caller must fall back to typing.
///
/// # Errors
/// Returns why it is unavailable, for `murmur doctor` to report verbatim.
pub fn availability() -> std::result::Result<(), String> {
    match paste::get_contents(
        ClipboardType::Regular,
        Seat::Unspecified,
        paste::MimeType::Text,
    ) {
        Ok(_) => Ok(()),
        // An empty clipboard still proves the protocol is there.
        Err(paste::Error::NoSeats | paste::Error::ClipboardEmpty | paste::Error::NoMimeType) => {
            Ok(())
        }
        Err(error) => Err(error.to_string()),
    }
}

#[must_use]
pub fn is_available() -> bool {
    availability().is_ok()
}
