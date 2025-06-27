use special_fun;
use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub enum WindowType {
    Rectangle,
    Sin,
    Vorbis,
    Hann,
    Blackman,
    BlackmanNuttall,
    KBD1,
    KBD5,
    KBD10,
}

impl WindowType {
    pub const ALL: [WindowType; 9] = [
        Self::Rectangle,
        Self::Sin,
        Self::Vorbis,
        Self::Hann,
        Self::Blackman,
        Self::BlackmanNuttall,
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
    pub fn generate_window(&self, window_size: usize) -> Vec<f32> {
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
            Self::Vorbis => {
                let mut window = vec![0.0f32; window_size];
                for n in 0..window_size {
                    let sqsin = f32::powf(((n as f32 + 0.5) * PI / window_size as f32).sin(), 2.0);
                    window[n] = (PI / 2.0 * sqsin).sin();
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
            Self::Blackman => {
                let mut window = vec![0.0f32; window_size];
                for n in 0..window_size / 2 {
                    window[n] = 7938.0 / 18608.0
                        - 9240.0 / 18608.0
                            * ((2.0 * n as f32 + 1.0) * PI / window_size as f32).cos()
                        + 1430.0 / 18608.0
                            * ((4.0 * n as f32 + 1.0) * PI / window_size as f32).cos();
                }
                for n in window_size / 2..window_size {
                    window[n] = window[window_size - 1 - n]
                }
                window
            }
            Self::BlackmanNuttall => {
                let mut window = vec![0.0f32; window_size];
                for n in 0..window_size / 2 {
                    window[n] = 0.3635819
                        - 0.4891775 * ((2.0 * n as f32 + 1.0) * PI / window_size as f32).cos()
                        + 0.1365995 * ((4.0 * n as f32 + 1.0) * PI / window_size as f32).cos()
                        - 0.0106411 * ((6.0 * n as f32 + 1.0) * PI / window_size as f32).cos();
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
            Self::Vorbis => "Vorbis",
            Self::Hann => "Hann",
            Self::Blackman => "Blackman",
            Self::BlackmanNuttall => "Blackman-Nuttall",
            Self::KBD1 => "KBD alpha=1",
            Self::KBD5 => "KBD alpha=5",
            Self::KBD10 => "KBD alpha=10",
        })
    }
}
