//! A presence in the panel, so the overlay can be got rid of and got back.
//!
//! Without this the overlay is a window with nowhere to go: minimise it and
//! there is no launcher entry, no panel icon and no titlebar to restore it from.
//! The tray is what makes it an application you can put away rather than one you
//! can only kill.
//!
//! GNOME dropped the legacy system tray, but implements `StatusNotifierItem`
//! through the AppIndicator extension that Ubuntu enables by default — which is
//! why this speaks that protocol directly rather than linking a toolkit.

use crate::{Message, icon};
use futures::channel::mpsc::Sender;
use std::sync::LazyLock;

/// Rendered once: the panel asks for the icon far more often than it changes.
static PANEL_ICON: LazyLock<ksni::Icon> = LazyLock::new(|| ksni::Icon {
    width: 64,
    height: 64,
    data: icon::argb(64, false),
});

pub struct Tray {
    to_interface: Sender<Message>,
    visible: bool,
    listening: bool,
}

impl Tray {
    pub fn new(to_interface: Sender<Message>) -> Self {
        Self { to_interface, visible: true, listening: false }
    }

    fn send(&mut self, message: Message) {
        let _ = self.to_interface.try_send(message);
    }
}

impl ksni::Tray for Tray {
    fn id(&self) -> String {
        "murmur".into()
    }

    fn title(&self) -> String {
        "Murmur".into()
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        vec![PANEL_ICON.clone()]
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip {
            title: "Murmur".into(),
            description: if self.listening {
                "listening".into()
            } else {
                "hold your trigger key and speak".into()
            },
            ..ksni::ToolTip::default()
        }
    }

    /// Left-clicking the panel icon toggles the overlay.
    ///
    /// The most common thing a user wants from a tray icon is the window back,
    /// and making them find it in a menu first is a wasted click.
    fn activate(&mut self, _x: i32, _y: i32) {
        self.visible = !self.visible;
        let message = if self.visible { Message::Show } else { Message::Hide };
        self.send(message);
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::{CheckmarkItem, MenuItem, StandardItem};

        vec![
            CheckmarkItem {
                label: "Show overlay".into(),
                checked: self.visible,
                activate: Box::new(|this: &mut Self| {
                    this.visible = !this.visible;
                    let message = if this.visible { Message::Show } else { Message::Hide };
                    this.send(message);
                }),
                ..CheckmarkItem::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Quit Murmur".into(),
                activate: Box::new(|this: &mut Self| this.send(Message::Quit)),
                ..StandardItem::default()
            }
            .into(),
        ]
    }
}

/// Publish the panel icon, and report what the user does with it.
///
/// Runs on its own thread with the blocking API: the tray must outlive every
/// individual dictation and has no business sharing the interface's executor.
/// A desktop with no `StatusNotifierWatcher` simply gets no icon — the overlay
/// still works, so this is a downgrade rather than a failure.
pub fn publish(to_interface: Sender<Message>) {
    use ksni::blocking::TrayMethods as _;

    std::thread::Builder::new()
        .name("murmur-tray".into())
        .spawn(move || match Tray::new(to_interface).spawn() {
            Ok(handle) => {
                tracing::info!("panel icon published");
                // The icon lives exactly as long as this thread and its handle:
                // returning here would drop the connection serving it, and the
                // icon would vanish moments after appearing.
                loop {
                    std::thread::park();
                    let _keep_alive = &handle;
                }
            }
            Err(error) => {
                tracing::warn!(%error, "no panel icon: this desktop has no status notifier");
            }
        })
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::channel::mpsc::channel;
    use ksni::Tray as _;

    fn tray() -> (Tray, futures::channel::mpsc::Receiver<Message>) {
        let (sender, receiver) = channel(8);
        (Tray::new(sender), receiver)
    }

    #[test]
    fn clicking_the_icon_toggles_the_overlay_both_ways() {
        let (mut tray, mut receiver) = tray();

        tray.activate(0, 0);
        assert!(matches!(receiver.try_recv(), Ok(Message::Hide)));

        tray.activate(0, 0);
        assert!(matches!(receiver.try_recv(), Ok(Message::Show)));
    }

    #[test]
    fn the_menu_offers_a_way_out_as_well_as_a_way_back() {
        let (tray, _receiver) = tray();
        let labels: Vec<String> = tray
            .menu()
            .into_iter()
            .filter_map(|item| match item {
                ksni::MenuItem::Standard(item) => Some(item.label),
                ksni::MenuItem::Checkmark(item) => Some(item.label),
                _ => None,
            })
            .collect();

        assert!(labels.iter().any(|l| l.contains("Show")), "{labels:?}");
        assert!(labels.iter().any(|l| l.contains("Quit")), "{labels:?}");
    }

    #[test]
    fn the_panel_icon_is_square_and_not_blank() {
        let (tray, _receiver) = tray();
        let icons = tray.icon_pixmap();
        let icon = icons.first().expect("an icon");
        assert_eq!(icon.width, icon.height);
        assert_eq!(icon.data.len(), (icon.width * icon.height * 4) as usize);
        assert!(icon.data.chunks_exact(4).any(|p| p[0] > 0), "the icon is fully transparent");
    }

    #[test]
    fn the_tooltip_says_what_it_is_doing() {
        let (mut tray, _receiver) = tray();
        assert!(tray.tool_tip().description.contains("speak"));

        tray.listening = true;
        assert!(tray.tool_tip().description.contains("listening"));
    }
}
