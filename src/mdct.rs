use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// FFTを用いたMDCT
pub fn mdct(input: &Vec<f32>, out: &mut Vec<f32>) {
    let num_samples = input.len();
    let m = num_samples / 2;
    let m2 = m / 2;
    let m4 = m / 4;
    let m32 = 3 * m / 2;
    let alpha = PI / (8.0 * m as f32);
    let omega = PI / m as f32;

    assert_eq!(m, out.len());

    // 回転因子
    let twiddle_factor = (0..m2)
        .map(|n| (Complex::I * (omega * n as f32 + alpha)).exp())
        .collect::<Vec<_>>();

    let mut fftspec = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        m2
    ];

    // 回転因子を乗じる
    // a[n] = -input[m32 - 1 - n] - input[m32 + n];     (n = 0, ..., m2 - 1)
    // a[n] =  input[n - m2]      - input[m32 - 1 - n]; (n = m2, ..., m - 1)
    // x[n] = (a[2n] - j * a[m - 1 - 2n]) * exp(j * (omega * n + alpha));
    for n in 0..m4 {
        let acomplex = Complex {
            re: -input[m32 - 1 - 2 * n] - input[m32 + 2 * n],
            im: input[m2 + 2 * n] - input[m2 - 1 - 2 * n],
        };
        fftspec[n] = acomplex * twiddle_factor[n];
    }
    for n in 0..m4 {
        let acomplex = Complex {
            re: input[2 * n] - input[m - 1 - 2 * n],
            im: input[m + 2 * n] + input[2 * m - 1 - 2 * n],
        };
        fftspec[m4 + n] = acomplex * twiddle_factor[m4 + n];
    }

    // m/2点でIFFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(m2);
    fft.process(&mut fftspec);

    // 回転因子を乗じる
    // x[n] *= exp(j * (omega * n + alpha));
    // output[n] = Real(x[n]);
    // output[m - 1 - n] = Imag(x[n]);
    for n in 0..m2 {
        let x = fftspec[n] * twiddle_factor[n];
        out[2 * n] = x.re;
        out[m - 1 - 2 * n] = x.im;
    }
}

/// IFFTを用いたMDCT
#[allow(dead_code)]
pub fn imdct(input: &Vec<f32>, out: &mut Vec<f32>) {
    let num_samples = 2 * input.len();
    let m = num_samples / 2;
    let m2 = m / 2;
    let m4 = m / 4;
    let m32 = 3 * m / 2;
    let m52 = 5 * m / 2;
    let alpha = PI / (8.0 * m as f32);
    let omega = PI / m as f32;

    assert_eq!(num_samples, out.len());

    // 回転因子
    let twiddle_factor = (0..m2)
        .map(|n| (Complex::I * (omega * n as f32 + alpha)).exp())
        .collect::<Vec<_>>();

    let mut fftspec = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        m2
    ];

    // FFT前に回転因子を乗じる
    // x[n] = (input[2n] - j * input[m - 1 - 2n]) * exp(j * (omega * n + alpha));
    for n in 0..m2 {
        let acomplex = Complex {
            re: input[2 * n],
            im: -input[m - 1 - 2 * n],
        };
        fftspec[n] = acomplex * twiddle_factor[n];
    }

    // m/2点でIFFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(m2);
    fft.process(&mut fftspec);

    // 回転因子を乗じ，結果出力
    for n in 0..m4 {
        let x = fftspec[n] * twiddle_factor[n];
        // DCT-IVのスペクトルをdct4[n]と書くと
        // dct4[2n]             = r1;
        // dct4[m - 2n - 1]     = i1;
        // output[m32 - 2n - 1] = -dct4[2n]         = -r1;
        // output[m32 + 2n]     = -dct4[2n]         = -r1;
        // output[m2 + n]       = -dct4[m - 2n - 1] = -i1;
        // output[m2 - n - 1]   =  dct4[m - 2n - 1] =  i1;
        out[m32 - 1 - 2 * n] = -x.re;
        out[m32 + 2 * n] = -x.re;
        out[m2 + 2 * n] = -x.im;
        out[m2 - 1 - 2 * n] = x.im;
    }
    for n in m4..m2 {
        let x = fftspec[n] * twiddle_factor[n];
        // DCT-IVのスペクトルをdct4[n]と書くと
        // dct4[2n]             = r1;
        // dct4[m - 2n - 1]     = i1;
        // output[m32 - 2n - 1] = -dct4[2n]         = -r1;
        // output[-m2 + 2n]     = dct4[2n]          =  r1;
        // output[m2 + n]       = -dct4[m - 2n - 1] = -i1;
        // output[m52 - n - 1]  = -dct4[m - 2n - 1] = -i1;
        out[m32 - 1 - 2 * n] = -x.re;
        out[2 * n - m2] = x.re;
        out[m2 + 2 * n] = -x.im;
        out[m52 - 1 - 2 * n] = -x.im;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ナイーブなMDCT
    fn naive_mdct(input: &Vec<f32>, out: &mut Vec<f32>) {
        let num_samples = input.len();
        let num_spectrum = num_samples / 2;

        assert_eq!(num_spectrum, out.len());

        for k in 0..num_spectrum {
            let mut tmp = 0.0;
            for n in 0..num_samples {
                tmp += input[n]
                    * (PI / num_spectrum as f32
                        * (k as f32 + 0.5)
                        * (n as f32 + 0.5 + num_spectrum as f32 / 2.0))
                        .cos();
            }
            out[k] = tmp;
        }
    }

    /// ナイーブなIMDCT
    fn naive_imdct(input: &Vec<f32>, out: &mut Vec<f32>) {
        let num_samples = 2 * input.len();
        let num_spectrum = num_samples / 2;

        assert_eq!(num_samples, out.len());

        for n in 0..num_samples {
            let mut tmp = 0.0;
            for k in 0..num_spectrum {
                tmp += input[k]
                    * (PI / num_spectrum as f32
                        * (k as f32 + 0.5)
                        * (n as f32 + 0.5 + num_spectrum as f32 / 2.0))
                        .cos();
            }
            out[n] = tmp;
        }
    }

    /// MDCTの一致確認
    fn check_mdct(input: &Vec<f32>, epsilon: f32) -> bool {
        let num_spectrum = input.len() / 2;
        let mut output_fft = vec![0.0f32; num_spectrum];
        let mut output_naive = vec![0.0f32; num_spectrum];

        mdct(&input, &mut output_fft);
        naive_mdct(&input, &mut output_naive);

        for k in 0..num_spectrum {
            if (output_fft[k] - output_naive[k]).abs() >= epsilon {
                return false;
            }
        }

        true
    }

    /// IMDCTの一致確認
    fn check_imdct(input: &Vec<f32>, epsilon: f32) -> bool {
        let num_samples = 2 * input.len();
        let mut output_fft = vec![0.0f32; num_samples];
        let mut output_naive = vec![0.0f32; num_samples];

        imdct(&input, &mut output_fft);
        naive_imdct(&input, &mut output_naive);

        for n in 0..num_samples {
            if (output_fft[n] - output_naive[n]).abs() >= epsilon {
                return false;
            }
        }

        true
    }

    #[test]
    fn test_mdct() {
        let epsilon = 1e-5;
        let num_samples = 16;
        let num_spectrum = 8;

        {
            let mut input = vec![0.0f32; num_samples];
            input[0] = 1.0f32;

            assert!(check_mdct(&input, epsilon));
        }

        {
            let input = (0..num_samples)
                .map(|x| x as f32 / num_samples as f32)
                .collect::<Vec<f32>>();

            assert!(check_mdct(&input, epsilon));
        }
    }

    #[test]
    fn test_imdct() {
        let epsilon = 1e-5;
        let num_samples = 16;
        let num_spectrum = 8;

        {
            let mut input = vec![0.0f32; num_spectrum];
            input[0] = 1.0f32;

            assert!(check_imdct(&input, epsilon));
        }

        {
            let input = (0..num_spectrum)
                .map(|x| x as f32 / num_spectrum as f32)
                .collect::<Vec<f32>>();

            assert!(check_imdct(&input, epsilon));
        }
    }
}
