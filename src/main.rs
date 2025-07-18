#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // Releaseビルドの時コンソールを非表示

use iced::{Font, Theme};

mod mdct;
mod wav_spectrum_viewer;
mod window_function;

use crate::wav_spectrum_viewer::WavSpectrumViewer;

pub fn main() -> iced::Result {
    iced::application(
        "WavSpectrogramViewer",
        WavSpectrumViewer::update,
        WavSpectrumViewer::view,
    )
    .subscription(WavSpectrumViewer::subscription)
    .font(include_bytes!("../fonts/icons.ttf").as_slice())
    .default_font(Font::MONOSPACE)
    .theme(|_| Theme::Dark)
    .antialiasing(true)
    .run_with(WavSpectrumViewer::new)
}
