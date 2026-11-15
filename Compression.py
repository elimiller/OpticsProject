# %% Compression of user-supplied TE0 temporal envelopes (local only)
"""Run cells in order. Populate INPUT_PROFILE in the empty data cell below.

Provide a COMPLEX field, not intensity. Physical convention: exp(-i omega t).
No cloud jobs. Grating settings and mode-table selection come from coupledmodemodel.
The grating is fixed for all inputs: its center period is based on that module's
CENTER_WAVELENGTH_UM, not on the carrier of the test pulse.
"""
from pathlib import Path
import importlib
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import coupledmodemodel as cmt

# Reload configuration after changes to coupledmodemodel.py in an Interactive kernel.
# This does not run its Gaussian example or submit any jobs.
importlib.reload(cmt)
PROJECT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUTPUT_DIR = PROJECT_DIR / 'outputs/compression'
MAX_OUTSIDE_ENERGY_FRACTION = 1e-4  # reject rather than silently clip significant energy

# %% Input format instructions
# In the EMPTY CELL below, assign INPUT_PROFILE using ONE of these formats.
# These are comments so that running the file without data does not fabricate input.
#
# Temporal ENVELOPE (not optical-carrier-resolved field), uniform time samples:
# INPUT_PROFILE = {
#     'name': 'nominal_input',
#     'time_s': time_array_in_seconds,
#     'field': complex_envelope_array,
#     'carrier_hz': cmt.c / (1.56e-6),
# }
#
# Spectral field, UNIFORM absolute frequency in Hz (not angular frequency):
# INPUT_PROFILE = {
#     'name': 'nominal_input',
#     'frequency_hz': absolute_frequency_array,
#     'spectrum': complex_spectral_field_array,
#     'carrier_hz': cmt.c / (1.56e-6),
# }
#
# For a CSV, use e.g. data = pd.read_csv(PROJECT_DIR / 'data' / 'input.csv')
# and field = data['real'].to_numpy() + 1j*data['imag'].to_numpy().
# Convert ps -> seconds by multiplying by 1e-12; THz -> Hz by 1e12.
# An amplitude/phase representation becomes amplitude*np.exp(1j*phase_radians).
# Wavelength-sampled spectra need conversion/resampling before this workflow;
# spectral densities per wavelength additionally need a Jacobian conversion.
# Inputs using exp(+i omega t) must be converted to the convention above.
# Use a unique name for each field; otherwise its saved results are overwritten.
# Multiple fields: define INPUT_PROFILES = [profile_a, profile_b, ...].

# %% YOUR INPUT DATA HERE (intentionally empty)

# %% Validation and Fourier transforms
def _uniform_arrays(axis, field):
    x = np.asarray(axis, dtype=float)
    a = np.asarray(field, dtype=complex)
    if x.ndim != 1 or a.shape != x.shape or len(x) < 16:
        raise ValueError('Provide matching 1D arrays with at least 16 samples.')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(a)):
        raise ValueError('Input contains nonfinite values.')
    order = np.argsort(x)
    x, a = x[order], a[order]
    dx = np.diff(x)
    if np.any(dx <= 0) or not np.allclose(dx, dx.mean(), rtol=1e-5, atol=0):
        raise ValueError('Axis must be unique and uniformly sampled. Resample first.')
    if np.max(abs(a)) == 0:
        raise ValueError('Input field is zero.')
    return x, a, dx.mean()


def prepare_spectrum(profile):
    """Forward transform uses +i Omega t; synthesis uses -i Omega t."""
    carrier = float(profile['carrier_hz'])
    if not np.isfinite(carrier) or carrier <= 0:
        raise ValueError('carrier_hz must be positive and finite.')
    temporal = 'time_s' in profile
    spectral = 'frequency_hz' in profile
    if temporal == spectral:
        raise ValueError('Provide either time_s/field OR frequency_hz/spectrum.')
    if temporal:
        time, field, dt = _uniform_arrays(profile['time_s'], profile['field'])
        # Do not assume the supplied time interval is centered on zero.
        offsets = np.fft.fftshift(np.fft.fftfreq(len(time), d=dt))
        spectrum = np.fft.fftshift(np.fft.ifft(field))*len(time)*dt
        spectrum *= np.exp(2j*np.pi*offsets*time[0])
        frequency = carrier + offsets
        edge = np.r_[abs(field[:3])**2, abs(field[-3:])**2].max()
        if edge > .001*np.max(abs(field)**2):
            warnings.warn('Temporal field reaches window edges; enlarge input time window.')
    else:
        frequency, spectrum, _ = _uniform_arrays(profile['frequency_hz'], profile['spectrum'])
    return frequency, spectrum, carrier


def intensity_fwhm(time_ps, intensity):
    """Width of peak-connected half-maximum interval, with interpolated crossings.

    This alone does not measure satellites/pedestals; inspect the full trace.
    """
    peak = int(np.argmax(intensity))
    half = intensity[peak]/2
    left = peak
    right = peak
    while left > 0 and intensity[left-1] >= half:
        left -= 1
    while right < len(intensity)-1 and intensity[right+1] >= half:
        right += 1
    if left == 0 or right == len(intensity)-1:
        return None
    tleft = np.interp(half, intensity[left-1:left+1], time_ps[left-1:left+1])
    tright = np.interp(half, intensity[right:right+2][::-1], time_ps[right:right+2][::-1])
    return float(tright-tleft)

# %% Fixed-grating compression function
def compress_profile(profile, output_dir=OUTPUT_DIR):
    frequency, spectrum, carrier = prepare_spectrum(profile)
    indices, low, high = cmt.load_indices(cmt.MODE_CSV)
    inside = (frequency >= low) & (frequency <= high)
    power = abs(spectrum)**2
    outside_fraction = float(power[~inside].sum()/power.sum())
    if outside_fraction > MAX_OUTSIDE_ENERGY_FRACTION:
        raise ValueError(f'{outside_fraction:.3%} input spectral energy lies outside '
                         'the mode tables. Extend the mode sweep before compression.')
    r = np.zeros(len(frequency), dtype=complex)
    if not inside.any():
        raise ValueError('No overlap with mode tables.')
    r[inside], _, period = cmt.reflection_spectrum(frequency[inside], *indices)
    # Out-of-table negligible tails are explicitly zeroed, never extrapolated.
    reflected = r*spectrum
    df = frequency[1]-frequency[0]
    count = len(frequency)
    times_ps = np.fft.fftshift(np.fft.fftfreq(count, d=df))*1e12
    def synthesize(a):
        # Lowest sampled frequency need not equal an FFT baseband bin.
        field = np.fft.fftshift(np.fft.fft(a))*df
        return field*np.exp(-2j*np.pi*(frequency[0]-carrier)*times_ps*1e-12)
    fields = [synthesize(spectrum), synthesize(reflected), synthesize(abs(reflected))]
    intensities = [abs(a)**2 for a in fields]
    norm = intensities[0].max()
    if norm == 0:
        raise ValueError('Input transform is zero.')
    for intensity in intensities:
        if max(intensity[:3].max(), intensity[-3:].max()) > .001*intensity.max():
            warnings.warn('A time trace reaches FFT boundaries: increase frequency '
                          'resolution/time window and check wraparound convergence.')
    name = str(profile.get('name', 'input_field'))
    if Path(name).name != name or name in ['', '.', '..']:
        raise ValueError('Profile name must be a simple folder name, not a path.')
    destination = Path(output_dir)/name
    destination.mkdir(parents=True, exist_ok=True)
    temporal = pd.DataFrame({'time_ps': times_ps})
    for label, field in zip(['input', 'output', 'output_transform_limit'], fields):
        temporal[f'{label}_real'] = field.real
        temporal[f'{label}_imag'] = field.imag
        temporal[f'{label}_intensity_over_input_peak'] = abs(field)**2/norm
    temporal.to_csv(destination/'temporal_fields.csv', index=False)
    pd.DataFrame(dict(frequency_hz=frequency, input_real=spectrum.real,
        input_imag=spectrum.imag, r_real=r.real, r_imag=r.imag,
        output_real=reflected.real, output_imag=reflected.imag,
        inside_mode_table=inside)).to_csv(destination/'spectral_fields.csv', index=False)
    metrics = dict(name=name, reflected_energy_fraction=float(abs(reflected @ reflected.conj())/
        power.sum()), outside_mode_table_energy_fraction=outside_fraction,
        input_fwhm_ps=intensity_fwhm(times_ps, intensities[0]),
        output_fwhm_ps=intensity_fwhm(times_ps, intensities[1]),
        output_transform_limit_fwhm_ps=intensity_fwhm(times_ps, intensities[2]),
        note='Uncalibrated scalar CMT; TE0 temporal envelope; arbitrary input units.',
        mode_csv=str(cmt.MODE_CSV), carrier_hz=carrier, period_center_um=period,
        length_um=cmt.LENGTH_UM, period_span_um=cmt.PERIOD_SPAN_UM,
        orientation=cmt.ORIENTATION, duty=cmt.DUTY, segments=cmt.SEGMENTS,
        kappa_scale=cmt.KAPPA_SCALE, loss_db_per_cm=cmt.LOSS_DB_PER_CM,
        apodization_fraction=cmt.APODIZATION_FRACTION)
    (destination/'metrics.json').write_text(json.dumps(metrics, indent=2))
    fig, ax = plt.subplots(figsize=(9, 4))
    for intensity, label in zip(intensities, ['Input', 'Reflected', 'Output transform limit']):
        ax.plot(times_ps, intensity/norm, label=label)
    ax.set(xlabel='Time (ps)', ylabel='Intensity / input peak', title=name)
    ax.legend(); fig.tight_layout()
    fig.savefig(destination/'compression.png', dpi=160)
    plt.show()
    print(json.dumps(metrics, indent=2))
    print(f'Saved: {destination}')
    return metrics

# %% Run after entering data; no invented field if the input cell is empty
if __name__ == '__main__':
    profiles = globals().get('INPUT_PROFILES')
    if profiles is None:
        single = globals().get('INPUT_PROFILE')
        profiles = [] if single is None else [single]
    if not profiles:
        print('No input loaded. Fill the YOUR INPUT DATA HERE cell, then rerun this cell.')
    else:
        compression_results = [compress_profile(profile) for profile in profiles]
