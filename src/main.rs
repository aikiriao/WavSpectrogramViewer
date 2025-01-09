#![windows_subsystem = "windows"]
use aifc;
use hound;
use iced::widget::canvas::{self, stroke, Cache, Canvas, Event, Frame, Geometry, Path, Stroke};
use iced::widget::{
    button, column, combo_box, container, horizontal_space, row, stack, text, text_input, tooltip,
};
use iced::{
    alignment, keyboard, Center, Color, Element, Fill, Font, Length, Point, Rectangle, Renderer,
    Size, Subscription, Task, Theme,
};
use iced::{event, mouse};

use iced::keyboard::key::Named;

use std::cmp;
use std::f32::consts::PI;
use std::ffi::OsStr;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use colorous::Gradient;
use realfft::RealFftPlanner;

mod mdct;
use crate::mdct::mdct;
use special_fun;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, PauseStreamError, PlayStreamError, Stream, StreamConfig};
use samplerate::{convert, ConverterType};

const YLABEL_WIDTH: f32 = 32.0;
const DEFAULT_MIN_HZ: f32 = 50.0;
const DEFAULT_MAX_HZ: f32 = 18000.0;
const WAVEFORM_HEIGHT_RATIO: f32 = 1.0 / 8.0;
const MIN_RANGE_RATIO_WIDTH: f64 = 1e-2;

pub fn main() -> iced::Result {
    iced::application(
        "WavSpectrumViewer - Iced",
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

#[derive(Debug, Clone)]
struct WavFormat {
    num_channels: u16,
    sampling_rate: f32,
    bits_per_sample: u16,
    num_samples_per_channel: usize,
}

#[derive(Debug, Clone)]
struct WavData {
    format: WavFormat,
    interleaved_pcm: Vec<f32>,
}

struct SpectrumLevelBar {
    cache: Cache,
    color_map: Option<ColorMap>,
}

struct PlayingPositionCursor {
    cache: Cache,
    position_ratio: Option<f32>,
}

struct WavSpectrumViewer {
    file: Option<PathBuf>,
    wav: Option<WavData>,
    cache: Cache,
    is_loading: bool,
    frame_size: Option<FrameSize>,
    frame_size_box: combo_box::State<FrameSize>,
    window_type: Option<WindowType>,
    window_type_box: combo_box::State<WindowType>,
    level_bar: SpectrumLevelBar,
    playing_cursor: PlayingPositionCursor,
    min_level_db: Option<LeveldB>,
    min_level_db_box: combo_box::State<LeveldB>,
    max_level_db: Option<LeveldB>,
    max_level_db_box: combo_box::State<LeveldB>,
    frequency_scale: Option<FrequencyScale>,
    frequency_scale_box: combo_box::State<FrequencyScale>,
    spectrum_view_mode: Option<SpectrumViewMode>,
    spectrum_view_mode_box: combo_box::State<SpectrumViewMode>,
    analyze_channel: Option<usize>,
    analyze_channel_box: combo_box::State<usize>,
    sample_range: Option<(usize, usize)>,
    sample_position: Option<usize>,
    hz_range: (f32, f32),
    hz_range_string: (String, String),
    stream_device: Device,
    stream_config: StreamConfig,
    stream: Option<Stream>,
    stream_is_playing: Arc<AtomicBool>,
    stream_played_samples: Arc<AtomicUsize>,
    stream_resampled_pcm: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
enum FrameSize {
    Size128,
    Size256,
    Size512,
    Size1024,
    Size2048,
    Size4096,
    Size8192,
}

#[derive(Debug, Clone)]
enum WindowType {
    Rectangle,
    Sin,
    Hann,
    KBD1,
    KBD5,
    KBD10,
}

#[derive(Debug, Clone)]
enum LeveldB {
    Level0dB,
    Level10dB,
    Level20dB,
    Level30dB,
    Level40dB,
    Level50dB,
    Level60dB,
    Level70dB,
    Level80dB,
    Level90dB,
    Level100dB,
    Level110dB,
    Level120dB,
    Level130dB,
}

#[derive(Debug, Clone)]
enum SpectrumViewMode {
    FFT,
    MDCT,
    FFTANDMDCT,
}

#[derive(Debug, Clone)]
enum ColorMap {
    TURBO,
    VIRIDIS,
    INFERNO,
    MAGMA,
    PLASMA,
    BLUES,
    REDS,
}

#[derive(Debug, Clone)]
enum FrequencyScale {
    Linear,
    Log,
    Bark,
}

// 選択範囲取得情報
#[derive(Debug, Clone)]
enum Pending {
    One { from: Point },
    Two { from: Point, to: Point },
}

// 描画範囲
#[derive(Debug, Clone)]
struct DrawRange {
    pending: Option<Pending>,
    range_ratio: (f64, f64),
}

#[derive(Debug, Clone)]
enum Message {
    OpenFile,
    FileOpened(Result<(PathBuf, WavData), Error>),
    FrameSizeSelected(FrameSize),
    WindowTypeSelected(WindowType),
    MinLeveldBSelected(LeveldB),
    MaxLeveldBSelected(LeveldB),
    MinHzInputed(String),
    MaxHzInputed(String),
    MinHzInputSubmited(String),
    MaxHzInputSubmited(String),
    ColorMapSelected(ColorMap),
    FrequencyScaleSelected(FrequencyScale),
    SpectrumViewModeSelected(SpectrumViewMode),
    AnalyzeChannelUpdated(usize),
    SampleRangeUpdated(Option<(usize, usize)>),
    CursorMovedOnSpectrum(Option<usize>),
    ReceivedPlayStartRequest,
    Tick,
    EventOccurred(iced::Event),
}

impl WavSpectrumViewer {
    fn new() -> (Self, Task<Message>) {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .expect("no output device available");

        (
            Self {
                file: None,
                wav: None,
                cache: Cache::default(),
                is_loading: false,
                frame_size: Some(FrameSize::Size512),
                frame_size_box: combo_box::State::new(FrameSize::ALL.to_vec()),
                window_type: Some(WindowType::Sin),
                window_type_box: combo_box::State::new(WindowType::ALL.to_vec()),
                level_bar: SpectrumLevelBar {
                    cache: Cache::default(),
                    color_map: Some(ColorMap::TURBO),
                },
                playing_cursor: PlayingPositionCursor {
                    cache: Cache::default(),
                    position_ratio: None,
                },
                min_level_db: Some(LeveldB::Level100dB),
                min_level_db_box: combo_box::State::new(LeveldB::ALL.to_vec()),
                max_level_db: Some(LeveldB::Level0dB),
                max_level_db_box: combo_box::State::new(LeveldB::ALL.to_vec()),
                frequency_scale: Some(FrequencyScale::Bark),
                frequency_scale_box: combo_box::State::new(FrequencyScale::ALL.to_vec()),
                spectrum_view_mode: Some(SpectrumViewMode::MDCT),
                spectrum_view_mode_box: combo_box::State::new(SpectrumViewMode::ALL.to_vec()),
                analyze_channel: Some(1),
                analyze_channel_box: combo_box::State::new(vec![1; 1]),
                sample_range: None,
                sample_position: None,
                stream_config: device.default_output_config().unwrap().into(),
                stream_device: device,
                hz_range: (DEFAULT_MIN_HZ, DEFAULT_MAX_HZ),
                hz_range_string: (DEFAULT_MIN_HZ.to_string(), DEFAULT_MAX_HZ.to_string()),
                stream: None,
                stream_is_playing: Arc::new(AtomicBool::new(false)),
                stream_played_samples: Arc::new(AtomicUsize::new(0)),
                stream_resampled_pcm: None,
            },
            Task::none(),
        )
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::OpenFile => {
                if self.is_loading {
                    Task::none()
                } else {
                    self.is_loading = true;

                    Task::perform(open_file(), Message::FileOpened)
                }
            }
            Message::FileOpened(result) => {
                self.is_loading = false;

                if let Ok((path, wav)) = result {
                    // 再生中の場合は止める
                    if self.stream_is_playing.load(Ordering::Relaxed) {
                        self.stream_play_stop().expect("Failed to stop play");
                    }

                    self.file = Some(path);
                    self.wav = Some(wav.clone());
                    self.sample_range = Some((0, wav.format.num_samples_per_channel - 1));
                    self.analyze_channel = Some(1);
                    self.analyze_channel_box =
                        combo_box::State::new((1..=wav.format.num_channels as usize).collect());
                    if let Some(stream) = &self.stream {
                        stream.pause().unwrap();
                    }
                    self.stream = None;
                    // 周波数範囲が波形のサンプリングレートより大きければクリップ
                    let nyquist_hz = wav.format.sampling_rate / 2.0;
                    if self.hz_range.1 > nyquist_hz {
                        self.hz_range.1 = nyquist_hz;
                    }
                    if self.hz_range.0 > nyquist_hz {
                        self.hz_range.0 = 0.0;
                    }
                    self.hz_range_string =
                        (self.hz_range.0.to_string(), self.hz_range.1.to_string());

                    // 出力先デバイスのレートに合わせてレート変換しておく
                    self.stream_resampled_pcm = Some(
                        convert(
                            wav.format.sampling_rate as u32,
                            self.stream_config.sample_rate.0 as u32,
                            wav.format.num_channels as usize,
                            ConverterType::SincBestQuality,
                            &wav.interleaved_pcm,
                        )
                        .unwrap(),
                    );
                    self.request_redraw();
                }

                Task::none()
            }
            Message::FrameSizeSelected(result) => {
                // 選択範囲に収まる範囲でフレームサイズを設定
                if let Some(range) = self.sample_range {
                    if (range.1 - range.0) > result.to_usize() {
                        self.frame_size = Some(result);
                        self.request_redraw();
                    }
                }

                Task::none()
            }
            Message::WindowTypeSelected(result) => {
                self.window_type = Some(result);
                self.request_redraw();

                Task::none()
            }
            Message::MinLeveldBSelected(result) => {
                // 最大値より小きい場合のみ設定
                if result.to_f32() < self.max_level_db.as_ref().unwrap().to_f32() {
                    self.min_level_db = Some(result);
                    self.request_redraw();
                }

                Task::none()
            }
            Message::MaxLeveldBSelected(result) => {
                // 最小値より大きい場合のみ設定
                if result.to_f32() > self.min_level_db.as_ref().unwrap().to_f32() {
                    self.max_level_db = Some(result);
                    self.request_redraw();
                }

                Task::none()
            }
            Message::MinHzInputed(result) => {
                if let Ok(_) = result.parse::<f32>() {
                    self.hz_range_string.0 = result;
                }

                Task::none()
            }
            Message::MaxHzInputed(result) => {
                if let Ok(_) = result.parse::<f32>() {
                    self.hz_range_string.1 = result;
                }

                Task::none()
            }
            Message::MinHzInputSubmited(result) => {
                // 最大値より小きい場合のみ設定
                if let Ok(hz) = result.parse::<f32>() {
                    if hz >= 0.0 && hz < self.hz_range.1 {
                        self.hz_range.0 = hz;
                        self.hz_range_string.0 = hz.to_string();
                        self.request_redraw();
                    } else {
                        // 範囲外の場合は設定値を戻す
                        self.hz_range_string.0 = self.hz_range.0.to_string();
                    }
                }

                Task::none()
            }
            Message::MaxHzInputSubmited(result) => {
                // 最小値より大きい場合のみ設定
                if let Some(wav) = &self.wav {
                    if let Ok(hz) = result.parse::<f32>() {
                        if hz <= wav.format.sampling_rate / 2.0 && hz > self.hz_range.0 {
                            self.hz_range.1 = hz;
                            self.hz_range_string.1 = hz.to_string();
                            self.request_redraw();
                        } else {
                            // 範囲外の場合は設定値を戻す
                            self.hz_range_string.1 = self.hz_range.1.to_string();
                        }
                    }
                }

                Task::none()
            }
            Message::ColorMapSelected(result) => {
                self.level_bar.color_map = Some(result.clone());
                self.request_redraw();
                self.level_bar.cache.clear();

                Task::none()
            }
            Message::FrequencyScaleSelected(result) => {
                self.frequency_scale = Some(result);
                self.request_redraw();

                Task::none()
            }
            Message::SpectrumViewModeSelected(result) => {
                self.spectrum_view_mode = Some(result);
                self.request_redraw();

                Task::none()
            }
            Message::AnalyzeChannelUpdated(result) => {
                self.analyze_channel = Some(result);
                self.request_redraw();

                Task::none()
            }
            Message::SampleRangeUpdated(result) => {
                if let Some(range) = self.sample_range {
                    if let Some(position) = self.sample_position {
                        // 変更前のサンプル位置比を保持
                        let offset = position - range.0;
                        let ratio = offset as f64 / (range.1 - range.0) as f64;
                        self.sample_range = result;
                        self.sample_position = self.get_sample_position_from_ratio(ratio);
                    } else {
                        self.sample_range = result;
                    }
                } else {
                    self.sample_range = result;
                }
                self.request_redraw();

                // 再生停止
                self.stream_play_stop().expect("Failed to stop play");

                Task::none()
            }
            Message::CursorMovedOnSpectrum(result) => {
                self.sample_position = result;

                Task::none()
            }
            Message::ReceivedPlayStartRequest => {
                if self.stream_is_playing.load(Ordering::Relaxed) {
                    // 再生中の場合は止める
                    self.stream_play_stop().expect("Failed to stop play");
                } else {
                    // 新規再生処理
                    self.stream_play_start().expect("Failed to start play");
                }

                Task::none()
            }
            Message::Tick => {
                if self.stream_is_playing.load(Ordering::Relaxed) {
                    // 再生位置を計算
                    let sampling_rate = self.wav.as_ref().unwrap().format.sampling_rate;
                    let num_channels = self.wav.as_ref().unwrap().format.num_channels as usize;
                    let played_samples =
                        self.stream_played_samples.load(Ordering::Relaxed) / num_channels;
                    // 再生済みサンプル数はレート変換が入っているので、レート変換比を計算
                    let rate_ratio = sampling_rate / self.stream_config.sample_rate.0 as f32;
                    let range_width =
                        (self.sample_range.unwrap().1 - self.sample_range.unwrap().0) as f32;
                    // 表示している範囲の再生位置を取得
                    let ratio = rate_ratio * played_samples as f32 / range_width;
                    self.playing_cursor.position_ratio = Some(ratio);
                    self.playing_cursor.cache.clear();
                } else {
                    // 再生終了を検知したら再生を止める
                    self.stream_play_stop().expect("Failed to stop play");
                }

                Task::none()
            }
            Message::EventOccurred(event) => match event {
                iced::event::Event::Window(event) => {
                    if let iced::window::Event::FileDropped(path) = event {
                        self.is_loading = true;
                        return Task::perform(load_file(path), Message::FileOpened);
                    }
                    Task::none()
                }
                _ => Task::none(),
            },
        }
    }

    fn view(&self) -> Element<Message> {
        let controls = row![
            action(
                open_icon(),
                "Open file",
                (!self.is_loading).then_some(Message::OpenFile)
            ),
            horizontal_space(),
            combo_box(
                &self.window_type_box,
                "Window type",
                self.window_type.as_ref(),
                Message::WindowTypeSelected
            ),
            combo_box(
                &self.frame_size_box,
                "Frame size",
                self.frame_size.as_ref(),
                Message::FrameSizeSelected
            ),
        ]
        .spacing(10)
        .align_y(Center);

        let spectrum_view = stack![
            Canvas::new(self).width(Fill).height(Fill),
            Canvas::new(&self.playing_cursor).width(Fill).height(Fill),
        ];

        let view_configures = row![
            text("CH"),
            combo_box(
                &self.analyze_channel_box,
                "CH",
                self.analyze_channel.as_ref(),
                Message::AnalyzeChannelUpdated
            )
            .width(20),
            combo_box(
                &self.spectrum_view_mode_box,
                "View mode",
                self.spectrum_view_mode.as_ref(),
                Message::SpectrumViewModeSelected
            ),
            combo_box(
                &self.frequency_scale_box,
                "Frequency scale",
                self.frequency_scale.as_ref(),
                Message::FrequencyScaleSelected
            ),
            text_input("Min hz", &self.hz_range_string.0)
                .on_input(Message::MinHzInputed)
                .on_submit(Message::MinHzInputSubmited(self.hz_range_string.0.clone())),
            text_input("Max hz", &self.hz_range_string.1)
                .on_input(Message::MaxHzInputed)
                .on_submit(Message::MaxHzInputSubmited(self.hz_range_string.1.clone())),
            text("min"),
            combo_box(
                &self.min_level_db_box,
                "Min level",
                self.min_level_db.as_ref(),
                Message::MinLeveldBSelected
            ),
            Canvas::new(&self.level_bar)
                .width(Fill)
                .height(Length::Fixed(30.0)),
            text("max"),
            combo_box(
                &self.max_level_db_box,
                "Max level",
                self.max_level_db.as_ref(),
                Message::MaxLeveldBSelected
            ),
        ]
        .spacing(10)
        .align_y(Center);

        let status = row![
            text(if let Some(path) = &self.file {
                let path = path.display().to_string();
                let path_len = path.chars().count();

                if path_len > 50 {
                    // パスが長い場合は省略
                    let slice_position = path.char_indices().nth(path_len - 30).unwrap().0;
                    format!("...{}", &path[slice_position..])
                } else {
                    path
                }
            } else {
                String::from("File not opened.")
            }),
            text(if let Some(wav) = &self.wav {
                format!(
                    "{:.1}kHz/{}samples",
                    wav.format.sampling_rate / 1000.0,
                    wav.format.num_samples_per_channel
                )
            } else {
                format!("")
            }),
            horizontal_space(),
            text(
                if self.stream.is_none() || !self.stream_is_playing.load(Ordering::Relaxed) {
                    format!("")
                } else {
                    // 再生位置を表示
                    let sampling_rate = self.wav.as_ref().unwrap().format.sampling_rate;
                    // レート変換の影響を加味
                    let rate_ratio = sampling_rate / self.stream_config.sample_rate.0 as f32;
                    let num_channels = self.wav.as_ref().unwrap().format.num_channels as usize;
                    // チャンネルあたりのサンプル数に変換
                    let played_samples =
                        self.stream_played_samples.load(Ordering::Relaxed) / num_channels;
                    let playing_position =
                        self.sample_range.unwrap().0 as f32 + played_samples as f32 * rate_ratio;
                    format!("{:.2}s", playing_position / sampling_rate)
                }
            ),
            text({
                if let Some(position) = self.sample_position {
                    let sampling_rate = self.wav.as_ref().unwrap().format.sampling_rate;
                    format!("{}({:.2}s)", position, position as f32 / sampling_rate)
                } else {
                    format!("")
                }
            }),
            text({
                if let Some(range) = self.sample_range {
                    let sampling_rate = self.wav.as_ref().unwrap().format.sampling_rate;
                    format!(
                        "{}({:.2}s):{}({:.2}s)",
                        range.0,
                        range.0 as f32 / sampling_rate,
                        range.1,
                        range.1 as f32 / sampling_rate
                    )
                } else {
                    format!("")
                }
            }),
        ]
        .spacing(10);

        column![controls, spectrum_view, view_configures, status,]
            .spacing(10)
            .padding(10)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        if self.stream_is_playing.load(Ordering::Relaxed) {
            iced::time::every(iced::time::Duration::from_millis(10)).map(|_| Message::Tick)
        } else {
            event::listen().map(Message::EventOccurred)
        }
    }

    // 再描画を依頼
    fn request_redraw(&self) {
        self.cache.clear();
    }

    // サンプル範囲比率からサンプル範囲を取得
    fn get_sample_range_from_ratio(&self, range_ratio: (f64, f64)) -> Option<(usize, usize)> {
        if let Some(wav) = &self.wav {
            let len = wav.format.num_samples_per_channel;
            let frame_size = self.frame_size.as_ref().unwrap().to_usize();
            Some(detect_sample_range(len, frame_size, range_ratio))
        } else {
            None
        }
    }

    // サンプル比率からサンプル位置を取得
    fn get_sample_position_from_ratio(&self, ratio: f64) -> Option<usize> {
        if let Some(range) = self.sample_range {
            let offset = range.0 as f64;
            let len = (range.1 - range.0) as f64;
            Some(f64::round(offset + len * ratio) as usize)
        } else {
            None
        }
    }

    // 再生開始
    fn stream_play_start(&mut self) -> Result<(), PlayStreamError> {
        if let Some(wav) = &self.wav {
            if let Some(resampled_pcm) = &self.stream_resampled_pcm {
                if let Some(range) = self.sample_range {
                    let num_channels = wav.format.num_channels as usize;
                    let sampling_rate = wav.format.sampling_rate;
                    // リサンプルした状態での範囲に変換
                    let resampled_range = (
                        f32::round(
                            range.0 as f32 * self.stream_config.sample_rate.0 as f32
                                / sampling_rate,
                        ) as usize,
                        f32::round(
                            range.1 as f32 * self.stream_config.sample_rate.0 as f32
                                / sampling_rate,
                        ) as usize,
                    );

                    let is_playing = self.stream_is_playing.clone();
                    let played_samples = self.stream_played_samples.clone();
                    let pcm: Vec<_> = resampled_pcm
                        [resampled_range.0 * num_channels..resampled_range.1 * num_channels]
                        .to_vec();

                    // 再生ストリーム作成
                    let stream = self
                        .stream_device
                        .build_output_stream(
                            &self.stream_config,
                            move |buffer: &mut [f32], _: &cpal::OutputCallbackInfo| {
                                let progress = played_samples.load(Ordering::Relaxed);
                                // 一旦バッファを無音で埋める
                                buffer.fill(0.0);
                                if progress < pcm.len() {
                                    // バッファにコピー
                                    let num_copy_samples =
                                        cmp::min(pcm.len() - progress, buffer.len());
                                    buffer[..num_copy_samples].copy_from_slice(
                                        &pcm[progress..progress + num_copy_samples],
                                    );
                                    // 再生サンプル増加
                                    played_samples
                                        .store(progress + num_copy_samples, Ordering::Relaxed);
                                } else {
                                    // 指定サンプル数を再生し終わった
                                    is_playing.store(false, Ordering::Relaxed);
                                }
                            },
                            |err| eprintln!("[WavSpectrumViewer] {err}"),
                            None,
                        )
                        .unwrap();

                    // 再生開始
                    self.stream_played_samples.store(0, Ordering::Relaxed);
                    self.stream_is_playing.store(true, Ordering::Relaxed);
                    stream.play()?;
                    self.stream = Some(stream);
                }
            }
        }

        Ok(())
    }

    // 再生停止
    fn stream_play_stop(&mut self) -> Result<(), PauseStreamError> {
        if let Some(stream) = &self.stream {
            self.stream_is_playing.store(false, Ordering::Relaxed);
            stream.pause()?;
            self.stream = None;
            // カーソル位置を消す
            self.playing_cursor.position_ratio = None;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum Error {
    DialogClosed,
    IoError(io::ErrorKind),
}

async fn open_file() -> Result<(PathBuf, WavData), Error> {
    let picked_file = rfd::AsyncFileDialog::new()
        .set_title("Open a wav or aiff file...")
        .pick_file()
        .await
        .ok_or(Error::DialogClosed)?;

    load_file(picked_file).await
}

async fn load_file(path: impl Into<PathBuf>) -> Result<(PathBuf, WavData), Error> {
    let path = path.into();

    if let Some(extension) = path.extension().and_then(OsStr::to_str) {
        match extension.to_lowercase().as_str() {
            "wav" => {
                let mut reader = hound::WavReader::open(&path).unwrap();
                let spec = reader.spec();
                let format = WavFormat {
                    num_channels: spec.channels,
                    sampling_rate: spec.sample_rate as f32,
                    bits_per_sample: spec.bits_per_sample,
                    num_samples_per_channel: reader.duration() as usize,
                };

                let num_samples = format.num_samples_per_channel * format.num_channels as usize;
                let max_pcm = (1 << (format.bits_per_sample - 1)) as f32;
                let norm_const = 1.0 / max_pcm;

                // チャンネルインターリーブで読みだす
                let mut pcm = vec![0.0; num_samples];
                for i in 0..num_samples {
                    let pcm_val = reader.samples::<i32>().next().unwrap().unwrap() as f32;
                    pcm[i] = pcm_val * norm_const;
                }

                return Ok((
                    path,
                    WavData {
                        format: format,
                        interleaved_pcm: pcm,
                    },
                ));
            }
            "aif" | "aiff" => {
                let mut stream = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
                let mut reader = aifc::AifcReader::new(&mut stream).unwrap();
                let info = reader.read_info().unwrap();
                let format = WavFormat {
                    num_channels: info.channels as u16,
                    sampling_rate: info.sample_rate as f32,
                    bits_per_sample: info.comm_sample_size as u16,
                    num_samples_per_channel: info.comm_num_sample_frames as usize,
                };

                let num_samples = format.num_samples_per_channel * format.num_channels as usize;
                let mut pcm = vec![0.0; num_samples];
                let max_pcm = (1 << (format.bits_per_sample - 1)) as f32;
                let norm_const = 1.0 / max_pcm;

                // チャンネルインターリーブで読みだす
                match info.sample_format {
                    aifc::SampleFormat::U8 | aifc::SampleFormat::I8 => {
                        let mut i = 0;
                        for sample in reader.samples().unwrap() {
                            if let Ok(aifc::Sample::I8(val)) = sample {
                                pcm[i] = val as f32 * norm_const;
                                i += 1;
                            }
                        }
                    }
                    aifc::SampleFormat::I16 | aifc::SampleFormat::I16LE => {
                        let mut i = 0;
                        for sample in reader.samples().unwrap() {
                            if let Ok(aifc::Sample::I16(val)) = sample {
                                pcm[i] = val as f32 * norm_const;
                                i += 1;
                            }
                        }
                    }
                    aifc::SampleFormat::I24 => {
                        let mut i = 0;
                        for sample in reader.samples().unwrap() {
                            if let Ok(aifc::Sample::I24(val)) = sample {
                                pcm[i] = val as f32 * norm_const;
                                i += 1;
                            }
                        }
                    }
                    aifc::SampleFormat::I32 | aifc::SampleFormat::I32LE => {
                        let mut i = 0;
                        for sample in reader.samples().unwrap() {
                            if let Ok(aifc::Sample::I32(val)) = sample {
                                pcm[i] = val as f32 * norm_const;
                                i += 1;
                            }
                        }
                    }
                    aifc::SampleFormat::F32 => {
                        let mut i = 0;
                        for sample in reader.samples().unwrap() {
                            if let Ok(aifc::Sample::F32(val)) = sample {
                                pcm[i] = val;
                                i += 1;
                            }
                        }
                    }
                    aifc::SampleFormat::F64 => {
                        let mut i = 0;
                        for sample in reader.samples().unwrap() {
                            if let Ok(aifc::Sample::F64(val)) = sample {
                                pcm[i] = val as f32;
                                i += 1;
                            }
                        }
                    }
                    _ => {
                        return Err(Error::IoError(io::ErrorKind::Unsupported));
                    }
                }

                return Ok((
                    path,
                    WavData {
                        format: format,
                        interleaved_pcm: pcm,
                    },
                ));
            }
            _ => {
                return Err(Error::IoError(io::ErrorKind::Unsupported));
            }
        }
    }

    return Err(Error::IoError(io::ErrorKind::Unsupported));
}

fn action<'a, Message: Clone + 'a>(
    content: impl Into<Element<'a, Message>>,
    label: &'a str,
    on_press: Option<Message>,
) -> Element<'a, Message> {
    let action = button(container(content).center_x(30));

    if let Some(on_press) = on_press {
        tooltip(
            action.on_press(on_press),
            label,
            tooltip::Position::FollowCursor,
        )
        .style(container::rounded_box)
        .into()
    } else {
        action.style(button::secondary).into()
    }
}

fn open_icon<'a, Message>() -> Element<'a, Message> {
    icon('\u{0f115}')
}

fn icon<'a, Message>(codepoint: char) -> Element<'a, Message> {
    const ICON_FONT: Font = Font::with_name("editor-icons");

    text(codepoint).font(ICON_FONT).into()
}

/// 波形描画
fn draw_waveform(frame: &mut Frame, bounds: Rectangle, pcm: &[f32]) {
    let center = bounds.center();
    let half_height = bounds.height / 2.0;
    let center_left = Point::new(center.x - bounds.width / 2.0, center.y);

    let num_points_to_draw = cmp::min(pcm.len(), 4 * bounds.width as usize); // 描画する点数（それ以外は間引く）
    let sample_stride = pcm.len() / num_points_to_draw;
    let x_offset_delta = bounds.width / num_points_to_draw as f32;

    // 描画する波形を拡大するため最大絶対値を計算
    let max_abs_pcm = pcm
        .iter()
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap()
        .abs();
    let pcm_normalizer = half_height / max_abs_pcm;

    // 背景を塗りつぶす
    frame.fill_rectangle(
        Point::new(bounds.x, bounds.y),
        Size::new(bounds.width, bounds.height),
        Color::from_rgb8(0, 0, 0),
    );

    let line_color = Color::from_rgb(0.0, 196.0, 0.0);
    let samples_per_pixel = pcm.len() as f32 / bounds.width;
    const USE_PATH_THRESHOLD: f32 = 200.0;
    if samples_per_pixel < USE_PATH_THRESHOLD {
        // 波形描画パスを生成
        let path = Path::new(|b| {
            b.move_to(Point::new(
                center_left.x,
                center.y + pcm[0] * pcm_normalizer,
            ));
            for i in 1..num_points_to_draw {
                b.line_to(Point::new(
                    center_left.x + i as f32 * x_offset_delta,
                    center.y + pcm[i * sample_stride] * pcm_normalizer,
                ));
            }
        });
        // 波形描画
        frame.stroke(
            &path,
            Stroke {
                style: stroke::Style::Solid(line_color),
                width: 1.0,
                ..Stroke::default()
            },
        );
    } else {
        // ピクセルあたりのサンプル数が多いときは、最小値と最大値をつなぐ矩形のみ描画
        let mut prev_sample = 0;
        for i in 0..num_points_to_draw {
            const MIN_HEIGHT: f32 = 0.5;
            let current_sample = (i + 1) * sample_stride;
            let max_val = pcm[prev_sample..current_sample]
                .iter()
                .max_by(|a, b| a.total_cmp(&b))
                .unwrap();
            let min_val = pcm[prev_sample..current_sample]
                .iter()
                .min_by(|a, b| a.total_cmp(&b))
                .unwrap();

            // 最大と最小の差がない（無音など）ときは高さをクリップ
            let mut height = (max_val - min_val) * pcm_normalizer;
            if height < MIN_HEIGHT {
                height = MIN_HEIGHT;
            }

            // 矩形描画
            frame.fill_rectangle(
                Point::new(
                    center_left.x + i as f32 * x_offset_delta,
                    center.y - max_val * pcm_normalizer,
                ),
                Size::new(1.0, height),
                line_color,
            );
            prev_sample = current_sample;
        }
    }
}

/// 線形周波数からBarkスケールに変換(Traunmuller 1990)
fn hz_to_bark(hz: f32) -> f32 {
    let bark = (26.81 * hz) / (1960.0 + hz) - 0.53;
    if bark < 2.0 {
        bark + 0.15 * (2.0 - bark)
    } else if bark > 20.1 {
        bark + 0.22 * (bark - 20.1)
    } else {
        bark
    }
}

/// Barkスケールから線形周波数スケールに変換(Traunmuller 1990)
fn bark_to_hz(mut bark: f32) -> f32 {
    if bark < 2.0 {
        bark = (bark - 0.3) / 0.85
    } else if bark > 20.1 {
        bark = (bark + 4.422) / 1.22
    }
    1960.0 * (0.53 + bark) / (26.28 - bark)
}

/// [0,1]の範囲に正規化
fn normalizer(val: f32, min: f32, max: f32) -> f32 {
    (val - min) / (max - min)
}

/// [0,1]の範囲から元に戻す
fn denormalizer(val: f32, min: f32, max: f32) -> f32 {
    val * (max - min) + min
}

/// ビンに対応する周波数(Hz)と正規化された位置を取得
fn get_bin_hz_poition(
    scale: &FrequencyScale,
    bin: f32,
    sampling_rate: f32,
    num_spectrum: usize,
    hz_range: (f32, f32),
) -> (f32, f32) {
    let hz = bin * sampling_rate as f32 / (2.0 * num_spectrum as f32);
    let norm_hz = match scale {
        FrequencyScale::Linear => normalizer(hz, hz_range.0, hz_range.1),
        FrequencyScale::Log => {
            let min_hz = if hz_range.0 <= 1.0 { 1.0 } else { hz_range.0 };
            if bin <= 0.0 {
                0.0
            } else {
                normalizer(hz.log10(), min_hz.log10(), hz_range.1.log10())
            }
        }
        FrequencyScale::Bark => normalizer(
            hz_to_bark(hz),
            hz_to_bark(hz_range.0),
            hz_to_bark(hz_range.1),
        ),
    };
    (hz, norm_hz)
}

/// 正規化された位置[0,1]から周波数を取得
fn get_hz_from_normalized_position(
    scale: &FrequencyScale,
    normalized_pos: f32,
    hz_range: (f32, f32),
) -> f32 {
    assert!(normalized_pos >= 0.0 && normalized_pos <= 1.0);
    match scale {
        FrequencyScale::Linear => denormalizer(normalized_pos, hz_range.0, hz_range.1),
        FrequencyScale::Log => {
            let min_hz = if hz_range.0 <= 1.0 { 1.0 } else { hz_range.0 };
            let aslog10 = denormalizer(normalized_pos, min_hz.log10(), hz_range.1.log10());
            f32::powf(10.0, aslog10)
        }
        FrequencyScale::Bark => {
            let asbark = denormalizer(
                normalized_pos,
                hz_to_bark(hz_range.0),
                hz_to_bark(hz_range.1),
            );
            bark_to_hz(asbark)
        }
    }
}

/// スペクトル描画
fn draw_spectrum(
    frame: &mut Frame,
    bounds: Rectangle,
    pcm: &[f32],
    sampling_rate: f32,
    num_frame_samples: usize,
    window_type: &WindowType,
    spectrum_maker: fn(&mut Vec<f32>, &mut Vec<f32>, usize),
    frequency_scale: FrequencyScale,
    hz_range: (f32, f32),
    db_range: (f32, f32),
    color_gradient: Gradient,
) {
    let num_spectrum = num_frame_samples / 2;

    let center = bounds.center();
    let height = bounds.height;
    let width = bounds.width;
    let spectrum_lower_left = Point::new(center.x - width / 2.0, center.y + height / 2.0);

    let mut spectrum = vec![0.0f32; num_spectrum];
    let window = window_type.generate_window(num_frame_samples);

    let num_frames_to_draw = width as usize / 2; // TODO: これはオーバーラップ幅に合わせるとよいかも
    let sample_frame_delta = if pcm.len() > num_frame_samples {
        (pcm.len() - num_frame_samples) as f32 / num_frames_to_draw as f32
    } else {
        1.0
    };
    let bin_range = (
        2.0 * hz_range.0 * num_spectrum as f32 / sampling_rate,
        2.0 * hz_range.1 * num_spectrum as f32 / sampling_rate,
    );
    let bin_length = f32::round(bin_range.1 - bin_range.0) as usize;
    let num_bins_to_draw = cmp::min(bin_length, height as usize); // 描画高がスペクトル数より多くならないようにクリップ
    let delta_bin = bin_length as f32 / num_bins_to_draw as f32;
    let min_spec_abs = f32::powf(10.0, db_range.0 / 20.0);

    let width_per_frame = width / num_frames_to_draw as f32;

    for fr in 0..num_frames_to_draw {
        let smpl_offset = (fr as f32 * sample_frame_delta) as usize;
        let mut input: Vec<f32> = pcm[smpl_offset..smpl_offset + num_frame_samples]
            .to_vec()
            .clone();
        // 窓かけ
        for n in 0..num_frame_samples {
            input[n] *= window[n];
        }
        // スペクトル生成
        spectrum_maker(&mut input, &mut spectrum, num_frame_samples);
        let logabs_spec = spectrum
            .iter()
            .map(|&x| {
                let abs = x.abs();
                if abs < min_spec_abs {
                    20.0 * min_spec_abs.log10()
                } else {
                    20.0 * abs.log10()
                }
            })
            .collect::<Vec<_>>();

        let mut hzpos = get_bin_hz_poition(
            &frequency_scale,
            bin_range.0 as f32,
            sampling_rate,
            num_spectrum,
            hz_range,
        );
        for i in 0..num_bins_to_draw {
            let bin = bin_range.0 + i as f32 * delta_bin;
            let next_hzpos = get_bin_hz_poition(
                &frequency_scale,
                bin + delta_bin,
                sampling_rate,
                num_spectrum,
                hz_range,
            );
            let spec = if (bin as usize) < (num_spectrum - 1) {
                // 隣接ビンと線形補間
                let bin_fraction = bin - bin.floor();
                bin_fraction * logabs_spec[bin as usize + 1]
                    + (1.0 - bin_fraction) * logabs_spec[bin as usize]
            } else {
                logabs_spec[bin as usize]
            };
            let color =
                color_gradient.eval_continuous(normalizer(spec, db_range.0, db_range.1) as f64);
            frame.fill_rectangle(
                Point::new(
                    spectrum_lower_left.x + width_per_frame * fr as f32,
                    spectrum_lower_left.y - next_hzpos.1 * bounds.height,
                ),
                Size::new(width_per_frame, (next_hzpos.1 - hzpos.1) * bounds.height),
                Color::from_rgb8(color.r, color.g, color.b),
            );
            hzpos = next_hzpos;
        }
    }
}

/// 周波数ラベル描画
fn draw_hzlabel(
    frame: &mut Frame,
    bounds: Rectangle,
    frequency_scale: FrequencyScale,
    hz_range: (f32, f32),
) {
    let num_hzlabel = 16;
    let hzlabel_right_x = bounds.center().x + bounds.width / 2.0;
    let hzlabel_bottom_y = bounds.center().y + bounds.height / 2.0;
    for i in 0..num_hzlabel {
        let normalized_pos = i as f32 / num_hzlabel as f32;
        let hz = get_hz_from_normalized_position(&frequency_scale, normalized_pos, hz_range);
        let hz_string = if hz < 1000.0 {
            format!("{:.0}", hz)
        } else {
            format!("{:.1}k", hz / 1000.0)
        };
        frame.fill_text(canvas::Text {
            content: hz_string,
            size: iced::Pixels(13.0),
            position: Point::new(
                hzlabel_right_x,
                hzlabel_bottom_y - normalized_pos * bounds.height,
            ),
            color: Color::WHITE,
            horizontal_alignment: alignment::Horizontal::Right,
            vertical_alignment: alignment::Vertical::Bottom,
            font: Font::MONOSPACE,
            ..canvas::Text::default()
        });
    }
}

/// 時刻ラベル描画
fn draw_timelabel(
    frame: &mut Frame,
    bounds: Rectangle,
    sampling_rate: f32,
    sample_range: (usize, usize),
) {
    let num_timelabel = 16;
    let timelabel_left_x = bounds.center().x - bounds.width / 2.0;
    let timelabel_y = bounds.center().y;
    let period = 1.0 / sampling_rate;
    for i in 0..num_timelabel {
        let normalized_pos = i as f32 / num_timelabel as f32;
        let sample = sample_range.0 as f32
            + (sample_range.1 as f32 - sample_range.0 as f32) * normalized_pos;
        let time = sample * period;
        frame.fill_text(canvas::Text {
            content: format!("{:.2}", time),
            size: iced::Pixels(13.0),
            position: Point::new(
                timelabel_left_x + normalized_pos * bounds.width,
                timelabel_y,
            ),
            color: Color::WHITE,
            horizontal_alignment: alignment::Horizontal::Left,
            vertical_alignment: alignment::Vertical::Bottom,
            font: Font::MONOSPACE,
            ..canvas::Text::default()
        });
    }
}

/// 振幅ラベル描画
fn draw_amplitude_label(frame: &mut Frame, bounds: Rectangle, pcm: &[f32]) {
    let label_right_x = bounds.center().x + bounds.width / 2.0 - 3.0;
    let label_bottom_y = bounds.center().y + bounds.height / 2.0;
    let max_abs_pcm = pcm
        .iter()
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap()
        .abs();

    let draw_amplitude_label = |frame: &mut Frame,
                                value: f32,
                                vertical_alignment: alignment::Vertical,
                                y_position: f32| {
        frame.fill_text(canvas::Text {
            content: format!("{:.2}", value),
            size: iced::Pixels(11.0),
            position: Point::new(label_right_x, y_position),
            color: Color::WHITE,
            horizontal_alignment: alignment::Horizontal::Right,
            vertical_alignment: vertical_alignment,
            font: Font::MONOSPACE,
            ..canvas::Text::default()
        });
    };

    // -最大振幅, 0, 最大振幅のみ描画
    draw_amplitude_label(
        frame,
        -max_abs_pcm,
        alignment::Vertical::Bottom,
        label_bottom_y,
    );
    draw_amplitude_label(frame, 0.0, alignment::Vertical::Center, bounds.center().y);
    draw_amplitude_label(
        frame,
        max_abs_pcm,
        alignment::Vertical::Top,
        label_bottom_y - bounds.height,
    );
}

/// サンプルの範囲を比率から計算
fn detect_sample_range(len: usize, frame_size: usize, range_ratio: (f64, f64)) -> (usize, usize) {
    let flen = len as f64;
    let start = (flen * range_ratio.0) as usize;
    let end = (flen * range_ratio.1) as usize;

    if start + frame_size < end {
        (start, end)
    } else {
        // 範囲が狭すぎる場合はフレームサイズに制限
        if start + frame_size < len {
            (start, start + frame_size + 1)
        } else {
            (end - frame_size - 1, end)
        }
    }
}

impl canvas::Program<Message> for WavSpectrumViewer {
    type State = Option<DrawRange>;

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        // wavが読み込まれていたら描画
        if let Some(wav) = &self.wav {
            let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
                let waveform_height = bounds.height * WAVEFORM_HEIGHT_RATIO;
                let spectrum_height = bounds.height - waveform_height;
                let frame_size = self.frame_size.as_ref().unwrap().to_usize();

                // 描画サンプル範囲
                let sample_range = if let Some(range) = self.sample_range {
                    range
                } else {
                    (
                        0,
                        self.wav.as_ref().unwrap().format.num_samples_per_channel - 1,
                    )
                };

                // 分析チャンネル・描画サンプル範囲の信号を取得
                let pcm = &(wav.interleaved_pcm[(self.analyze_channel.unwrap() - 1)..]
                    .to_vec()
                    .into_iter()
                    .step_by(wav.format.num_channels as usize)
                    .collect::<Vec<f32>>())[sample_range.0..sample_range.1];

                let sampling_rate = self.wav.as_ref().unwrap().format.sampling_rate;

                // 波形描画
                draw_waveform(
                    frame,
                    Rectangle::new(
                        Point::new(YLABEL_WIDTH, 0.0),
                        Size::new(bounds.width, waveform_height),
                    ),
                    &pcm,
                );

                // 時刻ラベル描画
                draw_timelabel(
                    frame,
                    Rectangle::new(
                        Point::new(YLABEL_WIDTH, 0.0),
                        Size::new(bounds.width, YLABEL_WIDTH),
                    ),
                    sampling_rate,
                    sample_range,
                );

                // 振幅ラベル描画
                draw_amplitude_label(
                    frame,
                    Rectangle::new(
                        Point::new(0.0, 0.0),
                        Size::new(YLABEL_WIDTH, waveform_height),
                    ),
                    &pcm,
                );

                // FFTスペクトルの生成
                let fft_make_spectrum =
                    |input: &mut Vec<f32>, spectrum: &mut Vec<f32>, num_frame_samples: usize| {
                        let mut fft_planner = RealFftPlanner::<f32>::new();
                        let r2c = fft_planner.plan_fft_forward(num_frame_samples);
                        let mut complex_spectrum = r2c.make_output_vec();
                        r2c.process(input, &mut complex_spectrum).unwrap();
                        let norm_factor = 1.0 / num_frame_samples as f32;
                        for n in 0..num_frame_samples / 2 {
                            spectrum[n] = complex_spectrum[n].norm() * norm_factor;
                        }
                    };

                // MDCTスペクトルの生成
                let mdct_make_spectrum =
                    |input: &mut Vec<f32>, spectrum: &mut Vec<f32>, num_frame_samples: usize| {
                        mdct(input, spectrum);
                        let norm_factor = 1.0 / num_frame_samples as f32;
                        for n in 0..num_frame_samples / 2 {
                            spectrum[n] *= norm_factor;
                        }
                    };

                let db_range = (
                    self.min_level_db.as_ref().unwrap().to_f32(),
                    self.max_level_db.as_ref().unwrap().to_f32(),
                );
                let frequency_scale = self.frequency_scale.as_ref().unwrap();

                // ビューあたりの高さを設定
                let spectrum_height_per_view =
                    if let Some(SpectrumViewMode::FFTANDMDCT) = self.spectrum_view_mode {
                        spectrum_height / 2.0
                    } else {
                        spectrum_height
                    };
                let spectrum_height_offset =
                    if let Some(SpectrumViewMode::FFTANDMDCT) = self.spectrum_view_mode {
                        spectrum_height_per_view
                    } else {
                        0.0
                    };

                // FFTスペクトル描画
                match self.spectrum_view_mode {
                    Some(SpectrumViewMode::FFT) | Some(SpectrumViewMode::FFTANDMDCT) => {
                        // 周波数ラベル描画
                        draw_hzlabel(
                            frame,
                            Rectangle::new(
                                Point::new(0.0, waveform_height),
                                Size::new(YLABEL_WIDTH, spectrum_height_per_view),
                            ),
                            frequency_scale.clone(),
                            self.hz_range,
                        );

                        // FFTスペクトル描画
                        draw_spectrum(
                            frame,
                            Rectangle::new(
                                Point::new(YLABEL_WIDTH, waveform_height),
                                Size::new(bounds.width, spectrum_height_per_view),
                            ),
                            &pcm,
                            sampling_rate,
                            frame_size,
                            self.window_type.as_ref().unwrap(),
                            fft_make_spectrum,
                            frequency_scale.clone(),
                            self.hz_range,
                            db_range,
                            self.level_bar.color_map.as_ref().unwrap().to_colorous(),
                        );
                    }
                    _ => {}
                }

                // MDCTスペクトル描画
                match self.spectrum_view_mode {
                    Some(SpectrumViewMode::MDCT) | Some(SpectrumViewMode::FFTANDMDCT) => {
                        // 周波数ラベル描画
                        draw_hzlabel(
                            frame,
                            Rectangle::new(
                                Point::new(0.0, waveform_height + spectrum_height_offset),
                                Size::new(YLABEL_WIDTH, spectrum_height_per_view),
                            ),
                            frequency_scale.clone(),
                            self.hz_range,
                        );

                        // MDCTスペクトル描画
                        draw_spectrum(
                            frame,
                            Rectangle::new(
                                Point::new(YLABEL_WIDTH, waveform_height + spectrum_height_offset),
                                Size::new(bounds.width, spectrum_height_per_view),
                            ),
                            &pcm,
                            sampling_rate,
                            frame_size,
                            self.window_type.as_ref().unwrap(),
                            mdct_make_spectrum,
                            frequency_scale.clone(),
                            self.hz_range,
                            db_range,
                            self.level_bar.color_map.as_ref().unwrap().to_colorous(),
                        );
                    }
                    _ => {}
                }
            });
            vec![geometry]
        } else {
            // ファイルがロードされるまでは何も描かない
            vec![]
        }
    }

    fn update(
        &self,
        state: &mut Self::State,
        event: Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> (iced::widget::canvas::event::Status, Option<Message>) {
        if state.is_none() {
            *state = Some(DrawRange {
                pending: None,
                range_ratio: (0.0, 1.0),
            });
        }
        // カーソル位置に関係しないキー操作関連のイベント処理
        match event {
            Event::Keyboard(keyboard::Event::KeyReleased {
                key: iced::keyboard::Key::Named(Named::Space),
                ..
            }) => {
                return (
                    iced::widget::canvas::event::Status::Captured,
                    Some(Message::ReceivedPlayStartRequest),
                );
            }
            _ => {}
        }
        // カーソル位置に関係するイベント処理
        if let Some(cursor_position) = cursor.position_in(bounds) {
            let pending_state = &mut state.as_mut().unwrap().pending;
            let get_ratio = |x: f32| {
                if x < YLABEL_WIDTH {
                    0.0
                } else {
                    (x - YLABEL_WIDTH) as f64 / (bounds.width - YLABEL_WIDTH) as f64
                }
            };
            match event {
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                    match pending_state {
                        None | Some(Pending::Two { .. }) => {
                            // 開始位置を取得
                            *pending_state = Some(Pending::One {
                                from: cursor_position,
                            });
                        }
                        _ => {}
                    }
                }
                Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                    match pending_state {
                        Some(Pending::One { from }) => {
                            // 終了位置を取得
                            *pending_state = Some(Pending::Two {
                                from: *from,
                                to: cursor_position,
                            });

                            if let Some(Pending::Two { from, to }) = pending_state {
                                // 画面内の比を取得
                                let mut from_ratio = get_ratio(from.x);
                                let mut to_ratio = get_ratio(to.x);
                                // from < toとなるように入れ替え
                                if from_ratio > to_ratio {
                                    (from_ratio, to_ratio) = (to_ratio, from_ratio)
                                }

                                // 注目範囲更新
                                let current_range = &mut state.as_mut().unwrap().range_ratio;
                                let range_width = current_range.1 - current_range.0;
                                if f64::abs(from_ratio - to_ratio) < MIN_RANGE_RATIO_WIDTH {
                                    // 範囲が狭すぎる場合、注目点の周りで倍のズーム
                                    let center_ratio = (from_ratio + to_ratio) / 2.0;
                                    let center = current_range.0 + center_ratio * range_width;
                                    current_range.0 = center - range_width / 4.0;
                                    current_range.1 = center + range_width / 4.0;
                                } else {
                                    // ドラッグ範囲で更新
                                    current_range.0 += from_ratio * range_width;
                                    current_range.1 -= (1.0 - to_ratio) * range_width;
                                }
                                // 端点を越えていたらクリップ
                                if current_range.0 < 0.0 {
                                    current_range.0 = 0.0;
                                }
                                if current_range.1 >= 1.0 {
                                    current_range.1 = 1.0;
                                }

                                return (
                                    iced::widget::canvas::event::Status::Captured,
                                    Some(Message::SampleRangeUpdated(
                                        self.get_sample_range_from_ratio(*current_range),
                                    )),
                                );
                            }
                        }
                        _ => {}
                    }
                }
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Right)) => {
                    if let Some(Pending::Two { .. }) = pending_state {
                        let current_range = &mut state.as_mut().unwrap().range_ratio;
                        let range_width = f64::abs(current_range.1 - current_range.0);
                        let center = (current_range.0 + current_range.1) / 2.0;
                        if range_width < MIN_RANGE_RATIO_WIDTH {
                            // 比率が近すぎる場合は固定値で幅を持たせる
                            current_range.0 = center - MIN_RANGE_RATIO_WIDTH;
                            current_range.1 = center + MIN_RANGE_RATIO_WIDTH;
                        } else {
                            // 幅が現在の2倍になるように修正
                            let cursor_center =
                                current_range.0 + get_ratio(cursor_position.x) * range_width;
                            current_range.0 = cursor_center - range_width;
                            current_range.1 = cursor_center + range_width;
                        }
                        // 端点を越えていたらクリップ
                        if current_range.0 < 0.0 {
                            current_range.0 = 0.0;
                        }
                        if current_range.1 >= 1.0 {
                            current_range.1 = 1.0;
                        }
                        return (
                            iced::widget::canvas::event::Status::Captured,
                            Some(Message::SampleRangeUpdated(
                                self.get_sample_range_from_ratio(*current_range),
                            )),
                        );
                    }
                }
                Event::Mouse(mouse::Event::CursorMoved { position }) => {
                    return (
                        iced::widget::canvas::event::Status::Captured,
                        Some(Message::CursorMovedOnSpectrum(
                            self.get_sample_position_from_ratio(get_ratio(position.x)),
                        )),
                    );
                }
                _ => {}
            }
        } else {
            // 範囲外に飛び出たときは位置情報を消す
            return (
                iced::widget::canvas::event::Status::Captured,
                Some(Message::CursorMovedOnSpectrum(None)),
            );
        }
        (iced::widget::canvas::event::Status::Captured, None)
    }
}

impl canvas::Program<Message> for SpectrumLevelBar {
    type State = usize;

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        const DRAW_LEVELS: usize = 100;
        let mut frame = Frame::new(renderer, bounds.size());
        let rect_width = bounds.width / DRAW_LEVELS as f32;
        let gradient = self.color_map.clone().unwrap().to_colorous();
        for i in 0..DRAW_LEVELS {
            let color = gradient.eval_rational(i, DRAW_LEVELS);
            frame.fill_rectangle(
                Point::new(rect_width * i as f32, 0.0),
                Size::new(rect_width, bounds.height),
                Color::from_rgb8(color.r, color.g, color.b),
            )
        }
        frame.fill_text(canvas::Text {
            content: format!("{}", self.color_map.as_ref().unwrap()),
            size: iced::Pixels(18.0),
            position: Point::new(bounds.width / 2.0, bounds.height / 2.0),
            color: Color::WHITE,
            horizontal_alignment: alignment::Horizontal::Center,
            vertical_alignment: alignment::Vertical::Center,
            font: Font::MONOSPACE,
            ..canvas::Text::default()
        });
        vec![frame.into_geometry()]
    }

    fn update(
        &self,
        state: &mut Self::State,
        event: Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> (iced::widget::canvas::event::Status, Option<Message>) {
        if let Some(_) = cursor.position_in(bounds) {
            match event {
                Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                    // カラーマップインデックスを更新
                    *state = (*state + 1) % ColorMap::ALL.len();
                    return (
                        iced::widget::canvas::event::Status::Captured,
                        Some(Message::ColorMapSelected(ColorMap::ALL[*state].clone())),
                    );
                }
                _ => {}
            }
        }
        (iced::widget::canvas::event::Status::Captured, None)
    }
}

impl canvas::Program<Message> for PlayingPositionCursor {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        if let Some(ratio) = self.position_ratio {
            let mut frame = Frame::new(renderer, bounds.size());
            let position_x = YLABEL_WIDTH + ratio * (bounds.width - YLABEL_WIDTH);
            let path = Path::new(|b| {
                b.move_to(Point::new(position_x, 0.0));
                b.line_to(Point::new(
                    position_x,
                    bounds.height * WAVEFORM_HEIGHT_RATIO,
                ));
            });

            // カーソル描画
            frame.stroke(
                &path,
                Stroke {
                    style: stroke::Style::Solid(Color::WHITE),
                    width: 1.0,
                    ..Stroke::default()
                },
            );

            vec![frame.into_geometry()]
        } else {
            vec![]
        }
    }
}

impl FrameSize {
    const ALL: [FrameSize; 7] = [
        Self::Size128,
        Self::Size256,
        Self::Size512,
        Self::Size1024,
        Self::Size2048,
        Self::Size4096,
        Self::Size8192,
    ];

    fn to_usize(&self) -> usize {
        match self {
            Self::Size128 => 128,
            Self::Size256 => 256,
            Self::Size512 => 512,
            Self::Size1024 => 1024,
            Self::Size2048 => 2048,
            Self::Size4096 => 4096,
            Self::Size8192 => 8192,
        }
    }
}

impl std::fmt::Display for FrameSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.to_usize()))
    }
}

impl WindowType {
    const ALL: [WindowType; 6] = [
        Self::Rectangle,
        Self::Sin,
        Self::Hann,
        Self::KBD1,
        Self::KBD5,
        Self::KBD10,
    ];

    /// パラメータalphaのKBD窓を生成
    fn generate_kbd_window(window_size: usize, alpha: f32) -> Vec<f32> {
        let mut window = vec![0.0f32; window_size];
        let mut sum = 0.0;
        for n in 0..window_size / 2 {
            let offset = 4.0 * n as f32 / window_size as f32 - 1.0;
            sum += special_fun::FloatSpecial::besseli(
                PI * alpha * (1.0 - f32::powf(offset, 2.0)).sqrt(),
                0.0,
            );
            window[n] = sum;
        }
        sum += special_fun::FloatSpecial::besseli(
            PI * alpha * (1.0 - f32::powf(8.0 / window_size as f32 - 1.0, 2.0)).sqrt(),
            0.0,
        );
        for n in 0..window_size / 2 {
            window[n] = (window[n] / sum).sqrt();
            window[window_size - n - 1] = window[n];
        }
        window
    }

    /// 窓を生成
    fn generate_window(&self, window_size: usize) -> Vec<f32> {
        match self {
            Self::Rectangle => {
                vec![1.0f32; window_size]
            }
            Self::Sin => {
                let mut window = vec![0.0f32; window_size];
                for n in 0..window_size {
                    window[n] = ((n as f32 + 0.5) * PI / window_size as f32).sin();
                }
                window
            }
            Self::Hann => {
                let mut window = vec![0.0f32; window_size];
                for n in 0..window_size / 2 {
                    window[n] =
                        0.5 * (1.0 - ((2.0 * n as f32 + 1.0) * PI / window_size as f32).cos());
                }
                for n in window_size / 2..window_size {
                    window[n] = window[window_size - 1 - n]
                }
                window
            }
            Self::KBD1 => Self::generate_kbd_window(window_size, 1.0),
            Self::KBD5 => Self::generate_kbd_window(window_size, 5.0),
            Self::KBD10 => Self::generate_kbd_window(window_size, 10.0),
        }
    }
}

impl std::fmt::Display for WindowType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Rectangle => "Rectangle",
            Self::Sin => "Sin",
            Self::Hann => "Hann",
            Self::KBD1 => "KBD alpha=1",
            Self::KBD5 => "KBD alpha=5",
            Self::KBD10 => "KBD alpha=10",
        })
    }
}

impl LeveldB {
    const ALL: [LeveldB; 14] = [
        Self::Level0dB,
        Self::Level10dB,
        Self::Level20dB,
        Self::Level30dB,
        Self::Level40dB,
        Self::Level50dB,
        Self::Level60dB,
        Self::Level70dB,
        Self::Level80dB,
        Self::Level90dB,
        Self::Level100dB,
        Self::Level110dB,
        Self::Level120dB,
        Self::Level130dB,
    ];

    fn to_f32(&self) -> f32 {
        match self {
            Self::Level0dB => 0.0,
            Self::Level10dB => -10.0,
            Self::Level20dB => -20.0,
            Self::Level30dB => -30.0,
            Self::Level40dB => -40.0,
            Self::Level50dB => -50.0,
            Self::Level60dB => -60.0,
            Self::Level70dB => -70.0,
            Self::Level80dB => -80.0,
            Self::Level90dB => -90.0,
            Self::Level100dB => -100.0,
            Self::Level110dB => -110.0,
            Self::Level120dB => -120.0,
            Self::Level130dB => -130.0,
        }
    }
}

impl std::fmt::Display for LeveldB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{} dB", self.to_f32()))
    }
}

impl ColorMap {
    const ALL: [ColorMap; 7] = [
        Self::TURBO,
        Self::VIRIDIS,
        Self::INFERNO,
        Self::MAGMA,
        Self::PLASMA,
        Self::BLUES,
        Self::REDS,
    ];

    fn to_colorous(&self) -> colorous::Gradient {
        match self {
            Self::TURBO => colorous::TURBO,
            Self::VIRIDIS => colorous::VIRIDIS,
            Self::INFERNO => colorous::INFERNO,
            Self::MAGMA => colorous::MAGMA,
            Self::PLASMA => colorous::PLASMA,
            Self::BLUES => colorous::BLUES,
            Self::REDS => colorous::REDS,
        }
    }
}

impl std::fmt::Display for ColorMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::TURBO => "TURBO",
            Self::VIRIDIS => "VIRIDIS",
            Self::INFERNO => "INFERNO",
            Self::MAGMA => "MAGMA",
            Self::PLASMA => "PLASMA",
            Self::BLUES => "BLUE",
            Self::REDS => "REDS",
        })
    }
}

impl FrequencyScale {
    const ALL: [FrequencyScale; 3] = [Self::Linear, Self::Log, Self::Bark];
}

impl std::fmt::Display for FrequencyScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Linear => "Linear",
            Self::Log => "Log",
            Self::Bark => "Bark",
        })
    }
}

impl SpectrumViewMode {
    const ALL: [SpectrumViewMode; 3] = [Self::FFT, Self::MDCT, Self::FFTANDMDCT];
}

impl std::fmt::Display for SpectrumViewMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::FFT => "FFT",
            Self::MDCT => "MDCT",
            Self::FFTANDMDCT => "FFT+MDCT",
        })
    }
}
