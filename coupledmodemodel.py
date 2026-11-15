# %% Local two-wave coupled-mode model for the LN grating
"""No Tidy3D/cloud calls. Run cells in order or execute this file.

Input: local mode CSVs for the narrow/wide uniform sections.
Output: complex TE0 reflection/transmission, delay/GDD, and Gaussian pulse test.
IMPORTANT: coupling is an UNCALIBRATED weak scalar effective-index estimate.
Geometry and neff alone do not determine exact vector interface scattering.
Validate uniform sections against EME before treating predictions as quantitative.

Convention: physical fields ~ exp(-i omega t), forward propagation exp(+i beta z).
Reference planes: entrance/exit of grating; no facet/circulator/external fiber.
Longitudinal coordinate s is independent of the mode solver's transverse z.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.interpolate import PchipInterpolator

# %% Configuration (all geometry lengths in micrometers)
PROJECT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
MODE_CSV = PROJECT_DIR / 'outputs/ln_bragg_modes/coarse_checking/all_modes.csv'
RUN_DIR = PROJECT_DIR / 'outputs/coupled_mode/nominal'
FILM_UM, ETCH_UM, SIDEWALL_DEG = 0.6, 0.3, 0.0
NARROW_UM, WIDE_UM = 1.1, 1.5
MODE_INDEX = 0  # inspect field profiles: mode index alone does not establish TE0
MIN_TE_FRACTION = 0.9
DUTY = 0.5  # fraction of each period occupied by WIDE section
CENTER_WAVELENGTH_UM = 1.56
LENGTH_UM = 1800.0
PERIOD_SPAN_UM = 0.008
ORIENTATION = 1  # +1 increasing period along input direction; -1 reversed
SEGMENTS = 1200  # repeat with 2x to check convergence of COMPLEX r / group delay
KAPPA_SCALE = 1.0  # fit against uniform-grating EME, then validate another length
LOSS_DB_PER_CM = 0.0  # phenomenological POWER loss, independent of numerical Im(neff)
APODIZATION_FRACTION = 0.1  # raised-cosine coupling ramp at each end; 0 = abrupt
# Phenomenological apodization: a fabricated width/duty ramp also changes detuning.
SAMPLES = 4096
TRANSFORM_LIMIT_FWHM_PS = 0.6  # intensity FWHM of nominal Gaussian before chirping
INPUT_GDD_PS2 = 1.0  # choose opposite grating GDD; change sign/orientation if needed
MIN_REFLECTANCE_FOR_PHASE = 0.05

# %% Input selection and interpolation (no extrapolation)
def load_indices(path):
    frame = pd.read_csv(path)
    mask = np.ones(len(frame), dtype=bool)
    for column, value in [('film_um', FILM_UM), ('etch_um', ETCH_UM),
                          ('sidewall_deg', SIDEWALL_DEG), ('mode_index', MODE_INDEX)]:
        mask &= np.isclose(frame[column], value)
    selected = frame[mask]
    sections = []
    for width in [NARROW_UM, WIDE_UM]:
        part = selected[np.isclose(selected.top_width_um, width)].sort_values('frequency_hz')
        if len(part) < 4 or part.frequency_hz.duplicated().any():
            raise ValueError(f'Need >=4 unique frequency samples for width {width}.')
        if (part.TE_fraction < MIN_TE_FRACTION).any():
            raise ValueError(f'Mode {MODE_INDEX} is not consistently TE-like at width {width}.')
        sections.append(part)
    low = max(p.frequency_hz.min() for p in sections)
    high = min(p.frequency_hz.max() for p in sections)
    if low >= high:
        raise ValueError('Mode tables have no shared frequency range.')
    return [PchipInterpolator(p.frequency_hz, p.neff_real, extrapolate=False)
            for p in sections], low, high

# %% Stable scalar scattering cascade
# d[A,B]/ds = [[i delta, i kappa],[-i kappa,-i delta]] [A,B].
# delta = beta_average - pi/period + i alpha_power/2.
# Segments use real kappa; S-matrices, rather than full transfer matrices, are cascaded.
def reflection_spectrum(frequency_hz, n_narrow, n_wide, *, length_um=LENGTH_UM,
                        period_center_um=None, period_span_um=PERIOD_SPAN_UM,
                        orientation=ORIENTATION, duty=DUTY, segments=SEGMENTS,
                        kappa_scale=KAPPA_SCALE, loss_db_per_cm=LOSS_DB_PER_CM,
                        apodization_fraction=APODIZATION_FRACTION):
    f = np.asarray(frequency_hz, dtype=float)
    if not (0 < duty < 1 and length_um > 0 and segments >= 1
            and orientation in [-1, 1] and 0 <= apodization_fraction <= .5
            and loss_db_per_cm >= 0):
        raise ValueError('Invalid grating parameters.')
    nn, nw = n_narrow(f), n_wide(f)
    if not np.all(np.isfinite(nn + nw)):
        raise ValueError('Requested frequencies exceed mode-table support.')
    lambda_m = c / f
    nbar = (1-duty)*nn + duty*nw
    # Square-wave Fourier coefficient under weak scalar effective-index approximation.
    kappa0 = kappa_scale * 2*(nw-nn)*np.sin(np.pi*duty)/lambda_m
    if period_center_um is None:
        f0 = c / (CENTER_WAVELENGTH_UM*1e-6)
        n0 = (1-duty)*float(n_narrow(f0)) + duty*float(n_wide(f0))
        period_center_um = CENTER_WAVELENGTH_UM/(2*n0)
    if period_center_um <= abs(period_span_um)/2:
        raise ValueError('Period must remain positive.')
    ds = length_um*1e-6/segments
    alpha = loss_db_per_cm*100*np.log(10)/10
    # Attach segments from far end toward entrance; no incident field at exit.
    r = np.zeros(f.shape, complex)
    t = np.ones(f.shape, complex)
    for j in range(segments-1, -1, -1):
        u = (j+.5)/segments
        period_m = (period_center_um + orientation*period_span_um*(u-.5))*1e-6
        ramp = 1.0
        if apodization_fraction > 0:
            edge = min(u, 1-u)/apodization_fraction
            if edge < 1:
                ramp = .5*(1-np.cos(np.pi*edge))
        kappa = kappa0*ramp
        delta = 2*np.pi*nbar/lambda_m - np.pi/period_m + .5j*alpha
        q = np.sqrt(kappa**2-delta**2+0j)
        # sinh(q ds)/q with its analytic limit at q=0.
        x = q*ds
        sinh_over_q = ds*np.sinc(1j*x/np.pi)
        denominator = np.cosh(x)-1j*delta*sinh_over_q
        local_r = 1j*kappa*sinh_over_q/denominator
        local_t = 1/denominator
        multiple_reflections = 1-local_r*r
        t = local_t*t/multiple_reflections
        r = local_r + local_t**2*r/multiple_reflections
    # Restore carrier-frame phase for transmission (reflection reference fixed at entrance).
    grating_phase = np.sum(np.pi / ((period_center_um + orientation*period_span_um*
                         ((np.arange(segments)+.5)/segments-.5))*1e-6))*ds
    return r, t*np.exp(1j*grating_phase), period_center_um


def phase_derivatives(f, r):
    """Derivatives are meaningful only where reflectance is non-negligible."""
    phase = np.unwrap(np.angle(r))
    omega = 2*np.pi*f
    delay = np.gradient(phase, omega, edge_order=2)*1e12
    gdd = np.gradient(delay, omega, edge_order=2)*1e12
    valid = abs(r)**2 >= MIN_REFLECTANCE_FOR_PHASE
    valid[:2] = valid[-2:] = False
    # Exclude derivative stencils crossing a low-reflection point.
    valid = np.convolve(valid.astype(int), np.ones(5, dtype=int), mode='same') == 5
    return phase, np.where(valid, delay, np.nan), np.where(valid, gdd, np.nan)

# %% Reflection spectrum on uniform frequency grid (required for subsequent FFT)
def run():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    indices, fmin, fmax = load_indices(MODE_CSV)
    f0 = c/(CENTER_WAVELENGTH_UM*1e-6)
    if not fmin < f0 < fmax:
        raise ValueError('Pulse carrier is outside the mode tables.')
    # Symmetric baseband, with f0 exactly on a bin; exclude duplicated FFT endpoint.
    halfspan = min(f0-fmin, fmax-f0)
    offsets = np.fft.fftshift(np.fft.fftfreq(SAMPLES, d=1/(2*halfspan)))
    freqs = f0+offsets
    r, t, period = reflection_spectrum(freqs, *indices)
    phase, delay, gdd = phase_derivatives(freqs, r)
    R, T = abs(r)**2, abs(t)**2
    spectrum = pd.DataFrame(dict(frequency_hz=freqs, wavelength_um=c/freqs*1e6,
        r_real=r.real, r_imag=r.imag, reflectance=R, t_real=t.real, t_imag=t.imag,
        transmittance=T, absorbed_or_lost_fraction=1-R-T,
        reflection_phase_rad=phase, group_delay_ps=delay, gdd_ps2=gdd))
    spectrum.to_csv(RUN_DIR/'reflection.csv', index=False)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    wavelength = c/freqs*1e6
    axes[0].plot(wavelength, R, label='R'); axes[0].plot(wavelength, T, label='T')
    axes[0].legend(); axes[0].set_ylabel('Power fraction')
    axes[1].plot(wavelength, delay); axes[1].set_ylabel('Group delay (ps)')
    axes[2].plot(wavelength, gdd); axes[2].set_ylabel('GDD (ps²)')
    axes[2].set_xlabel('Wavelength (µm)')
    fig.suptitle('Uncalibrated scalar coupled-mode model')
    fig.tight_layout(); fig.savefig(RUN_DIR/'reflection.png', dpi=160); plt.close(fig)

    # Nominal Gaussian pulse, physical synthesis exp(-i Omega t)
    omega = 2*np.pi*offsets
    tau = TRANSFORM_LIMIT_FWHM_PS*1e-12
    amplitude = np.exp(-omega**2*tau**2/(8*np.log(2)))
    if max(amplitude[0], amplitude[-1]) > 1e-3:
        warnings.warn('Pulse spectrum reaches simulation edges; widen the MODE sweep.')
    input_spectrum = amplitude*np.exp(.5j*INPUT_GDD_PS2*1e-24*omega**2)
    df = freqs[1]-freqs[0]
    times = np.fft.fftshift(np.fft.fftfreq(SAMPLES, d=df))*1e12
    def synthesize(a):
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(a)))*df
    ain, aout = synthesize(input_spectrum), synthesize(input_spectrum*r)
    atl = synthesize(abs(input_spectrum*r))  # transform limit of SAME output spectrum
    intensities = [abs(a)**2 for a in (ain, aout, atl)]
    norm = intensities[0].max()
    pd.DataFrame(dict(time_ps=times, input_intensity=intensities[0]/norm,
        output_intensity=intensities[1]/norm,
        output_transform_limit_intensity=intensities[2]/norm)).to_csv(
            RUN_DIR/'pulse.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for intensity, label in zip(intensities, ['Input', 'Reflected', 'Output transform limit']):
        ax.plot(times, intensity/norm, label=label)
    view = max(10, 6*tau*1e12*np.sqrt(1+(4*np.log(2)*INPUT_GDD_PS2/
                                                    TRANSFORM_LIMIT_FWHM_PS**2)**2))
    ax.set(xlim=(-view, view), xlabel='Time (ps)', ylabel='Intensity / input peak')
    ax.legend(); fig.tight_layout(); fig.savefig(RUN_DIR/'pulse.png', dpi=160); plt.close(fig)
    metadata = dict(mode_csv=str(MODE_CSV), model='UNCALIBRATED scalar two-wave CMT',
        film_um=FILM_UM, etch_um=ETCH_UM, sidewall_deg=SIDEWALL_DEG,
        narrow_um=NARROW_UM, wide_um=WIDE_UM, mode_index=MODE_INDEX,
        period_center_um=period, period_span_um=PERIOD_SPAN_UM, length_um=LENGTH_UM,
        orientation=ORIENTATION, duty=DUTY, segments=SEGMENTS, kappa_scale=KAPPA_SCALE,
        loss_db_per_cm=LOSS_DB_PER_CM, apodization_fraction=APODIZATION_FRACTION,
        input_gdd_ps2=INPUT_GDD_PS2, transform_limit_fwhm_ps=TRANSFORM_LIMIT_FWHM_PS,
        carrier_wavelength_um=CENTER_WAVELENGTH_UM, samples=SAMPLES,
        reflected_energy_fraction=float(np.sum(abs(input_spectrum*r)**2)/
                                        np.sum(abs(input_spectrum)**2)),
        max_power_balance_error=float(np.max(abs(1-R-T))) if LOSS_DB_PER_CM == 0 else None)
    (RUN_DIR/'configuration.json').write_text(json.dumps(metadata, indent=2))
    print(f'Estimated center period: {period*1000:.2f} nm. Results: {RUN_DIR}')
    print(f'Reflected energy fraction: {metadata["reflected_energy_fraction"]:.3f}')
    print('Not optimized/calibrated. Check sign of delay/GDD before selecting input chirp.')
    return spectrum, metadata

# %% Execute only when running this file/cell sequence, not when importing for tests
if __name__ == '__main__':
    spectrum, metadata = run()
