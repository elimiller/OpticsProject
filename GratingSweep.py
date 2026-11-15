# %% Local parameter maps for the fixed 1565-nm input comb
"""Run all cells or execute this file. No mode solves or cloud jobs.

Each map is a 2D slice: grating length versus signed total period span.
Positive span = period increases along input direction; negative = reversed.
The center period, section widths, duty cycle, and coupling model stay fixed.
All results are predictions of UNCALIBRATED scalar coupled-mode theory.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Compression as compression

cmt = compression.cmt
PROJECT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUTPUT_DIR = PROJECT_DIR / 'outputs/grating_sweep/length_chirp'
LENGTHS_MM = np.linspace(0.5, 3.5, 13)
SIGNED_PERIOD_SPANS_NM = np.linspace(-16, 16, 25)
CONCENTRATION_WINDOW_PS = 1.0  # total width, NOT +/-1 ps
MIN_REFLECTED_ENERGY_FOR_RANKING = 0.5
SEGMENTS = 1200
FFT_OVERSAMPLING = 8  # interpolate periodic waveform; adds no spectral information
# Use the same exact CSV coefficients, with negligible out-of-table tails zeroed.

# %% Periodic pulse metrics: robust to peaks crossing the time-window boundary
def periodic_metrics(intensity, dt_ps, window_ps=CONCENTRATION_WINDOW_PS):
    intensity = np.asarray(intensity, dtype=float)
    energy = intensity.sum()
    if not np.isfinite(energy) or energy <= 0:
        return dict(fwhm_ps=np.nan, pulse_concentration=np.nan)
    n = len(intensity)
    peak = int(np.argmax(intensity))
    centered = np.roll(intensity, n//2-peak)
    half = centered[n//2]/2
    left = right = n//2
    while left > 0 and centered[left-1] >= half:
        left -= 1
    while right < n-1 and centered[right+1] >= half:
        right += 1
    if left == 0 or right == n-1:
        width = np.nan  # no isolated half-maximum crossings (e.g. CW)
    else:
        l = left-1 + (half-centered[left-1])/(centered[left]-centered[left-1])
        r = right + (centered[right]-half)/(centered[right]-centered[right+1])
        width = (r-l)*dt_ps
    relative_time = (np.arange(n)-n//2)*dt_ps
    mask = abs(relative_time) <= window_ps/2
    concentration = centered[mask].sum()/energy
    return dict(fwhm_ps=float(width), pulse_concentration=float(concentration))


def temporal_intensity(spectrum, oversampling=FFT_OVERSAMPLING):
    # Ascending contiguous frequencies. FFT is physical exp(-i Omega t) synthesis.
    # Zero padding extends the high-frequency end; carrier phase cancels in |A|².
    return abs(np.fft.fftshift(np.fft.fft(spectrum, n=len(spectrum)*oversampling)))**2

# %% Sweep and plots
# Functions stay together in cells: no cell markers inside function bodies.
def run_sweep():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = compression.INPUT_PROFILE
    frequency, field, carrier = compression.prepare_spectrum(profile)
    df = float(np.mean(np.diff(frequency)))
    period_ps = 1e12/df
    if not 0 < CONCENTRATION_WINDOW_PS < period_ps:
        raise ValueError('Concentration window must lie within one repetition period.')
    indices, low, high = cmt.load_indices(cmt.MODE_CSV)
    inside = (frequency >= low) & (frequency <= high)
    input_energy = np.sum(abs(field)**2)
    omitted = float(np.sum(abs(field[~inside])**2)/input_energy)
    if omitted > compression.MAX_OUTSIDE_ENERGY_FRACTION:
        raise ValueError('Input energy exceeds mode-table coverage; extend local modes.')
    f0 = cmt.c/(cmt.CENTER_WAVELENGTH_UM*1e-6)
    n0 = (1-cmt.DUTY)*float(indices[0](f0))+cmt.DUTY*float(indices[1](f0))
    center_period = cmt.CENTER_WAVELENGTH_UM/(2*n0)
    if not np.isfinite(center_period):
        raise ValueError('Grating center lies outside mode tables.')
    dt_ps = period_ps/(len(field)*FFT_OVERSAMPLING)
    input_intensity = temporal_intensity(field)
    transform_limit = temporal_intensity(abs(field))
    rows = []
    total = len(LENGTHS_MM)*len(SIGNED_PERIOD_SPANS_NM)
    best = None
    def evaluate(length_mm, signed_span_nm, segments):
        r = np.zeros(len(field), complex)
        r[inside], _, _ = cmt.reflection_spectrum(frequency[inside], *indices,
            length_um=length_mm*1000, period_center_um=center_period,
            period_span_um=abs(signed_span_nm)*1e-3,
            orientation=1 if signed_span_nm >= 0 else -1, segments=segments)
        out = field*r
        intensity = temporal_intensity(out)
        metrics = periodic_metrics(intensity, dt_ps)
        metrics['reflected_energy_fraction'] = float(np.sum(abs(out)**2)/input_energy)
        return r, intensity, metrics
    for length in LENGTHS_MM:
        for span in SIGNED_PERIOD_SPANS_NM:
            r, intensity, metrics = evaluate(float(length), float(span), SEGMENTS)
            row = dict(length_mm=float(length), signed_period_span_nm=float(span), **metrics)
            rows.append(row)
            eligible = metrics['reflected_energy_fraction'] >= MIN_REFLECTED_ENERGY_FOR_RANKING
            if eligible and (best is None or metrics['pulse_concentration'] > best['pulse_concentration']):
                best = row.copy()
        # Checkpoint every completed length so an interrupted run retains progress.
        pd.DataFrame(rows).to_csv(OUTPUT_DIR/'metrics.csv', index=False)
        print(f'Completed {len(rows)}/{total} designs (L={length:.2f} mm)', flush=True)
    table = pd.DataFrame(rows)
    specs = [('fwhm_ps', 'Peak-connected intensity FWHM (ps)', 'viridis_r', None),
             ('reflected_energy_fraction', 'Reflected energy / input energy', 'viridis', (0, 1)),
             ('pulse_concentration', f'Reflected energy within {CONCENTRATION_WINDOW_PS:g} ps '
                                    'of peak / total reflected energy', 'viridis', (0, 1))]
    for column, title, cmap, limits in specs:
        matrix = table.pivot(index='length_mm', columns='signed_period_span_nm', values=column)
        fig, ax = plt.subplots(figsize=(9, 5))
        mesh = ax.pcolormesh(matrix.columns, matrix.index, np.ma.masked_invalid(matrix.values),
            shading='nearest', cmap=cmap, **({} if limits is None else dict(vmin=limits[0], vmax=limits[1])))
        fig.colorbar(mesh, ax=ax)
        if best is not None:
            ax.plot(best['signed_period_span_nm'], best['length_mm'], 'rx', ms=9,
                    label=f'Highest concentration with energy ≥ {MIN_REFLECTED_ENERGY_FOR_RANKING:.0%}')
            ax.legend(fontsize=8)
        ax.set(xlabel='Signed total period span (nm); negative = reversed grating',
               ylabel='Grating length (mm)', title=title)
        fig.text(.5, .015, 'Fixed 1565-nm center, duty=0.5; uncalibrated scalar CMT',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(OUTPUT_DIR/f'{column}_heatmap.png', dpi=180)
        plt.show(); plt.close(fig)
    info = dict(input_csv=profile.get('source_csv'), mode_csv=str(cmt.MODE_CSV),
        coefficient_convention=profile.get('coefficient_convention'),
        lengths_mm=LENGTHS_MM.tolist(), signed_period_spans_nm=SIGNED_PERIOD_SPANS_NM.tolist(),
        center_wavelength_um=cmt.CENTER_WAVELENGTH_UM, center_period_um=center_period,
        narrow_width_um=cmt.NARROW_UM, wide_width_um=cmt.WIDE_UM,
        film_um=cmt.FILM_UM, etch_um=cmt.ETCH_UM, sidewall_deg=cmt.SIDEWALL_DEG,
        duty=cmt.DUTY, kappa_scale=cmt.KAPPA_SCALE, loss_db_per_cm=cmt.LOSS_DB_PER_CM,
        apodization_fraction=cmt.APODIZATION_FRACTION, segments=SEGMENTS,
        concentration_window_ps=CONCENTRATION_WINDOW_PS,
        ranking_min_reflected_energy=MIN_REFLECTED_ENERGY_FOR_RANKING,
        fft_oversampling=FFT_OVERSAMPLING, repetition_period_ps=period_ps,
        outside_mode_table_energy_fraction=omitted,
        input_metrics=periodic_metrics(input_intensity, dt_ps),
        input_transform_limit_metrics=periodic_metrics(transform_limit, dt_ps),
        best_sampled_design=best, note='Finite 2D sweep; not a global optimum. '
        'No radiation/fabrication loss unless modeled phenomenologically. '
        'FWHM can be narrower than transform-limited FWHM with larger pedestals.')
    if best is not None:
        r, intensity, _ = evaluate(best['length_mm'], best['signed_period_span_nm'], SEGMENTS)
        r2, intensity2, refined = evaluate(best['length_mm'], best['signed_period_span_nm'], 2*SEGMENTS)
        info['best_design_2x_segments'] = dict(**refined,
            max_complex_reflection_difference=float(np.max(abs(r-r2))))
        time = (np.arange(len(intensity))-len(intensity)//2)*dt_ps
        fig, ax = plt.subplots(figsize=(9, 4))
        # Align peaks for shape comparison; absolute arrival times are intentionally removed.
        for curve, label in [(input_intensity, 'Input'), (intensity, 'Best sampled design'),
                             (intensity2, 'Same design, 2× segments'),
                             (transform_limit, 'Input-spectrum transform limit')]:
            centered = np.roll(curve, len(curve)//2-int(np.argmax(curve)))
            ax.plot(time, centered/input_intensity.max(), label=label)
        ax.set(xlabel='Time relative to each peak (ps)', ylabel='Intensity / input peak',
               title='Pulse shapes over one repetition period (peaks aligned)')
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(OUTPUT_DIR/'best_design_pulses.png', dpi=180)
        plt.show(); plt.close(fig)
    (OUTPUT_DIR/'configuration_and_best.json').write_text(json.dumps(info, indent=2))
    print(f'Saved maps and metrics: {OUTPUT_DIR}')
    print('Best sampled design:', best)
    return table, info

# %% Execute the local analytical sweep
if __name__ == '__main__':
    sweep_metrics, sweep_info = run_sweep()
