# %% Refine the coarse length/chirp search without any new mode or cloud solves
"""Run all cells or execute this file.

Reuses GratingSweep's CMT evaluation, periodic pulse metrics, plotting, and
2x-segment convergence check. Ranks by energy concentration inside a fixed
1-ps window, subject to the same minimum reflected energy as the coarse search.
This is a LOCAL refinement of a fixed-center, fixed-cross-section 2D slice,
not a global optimum and not electromagnetic validation.
"""
from pathlib import Path
import json
import numpy as np
import GratingSweep as sweep

PROJECT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
COARSE_RESULTS = PROJECT_DIR / 'outputs/grating_sweep/length_chirp/configuration_and_best.json'
# Boundary expansion from initial 1.75–2.75 mm refinement (winner was at 1.75 mm).
REFINED_OUTPUT = PROJECT_DIR / 'outputs/grating_sweep/refined_length_chirp'
LENGTH_RANGE_MM = (0.5, 2.75)  # expanded after the first refinement hit its boundary
SIGNED_SPAN_RANGE_NM = (-40/3, -4/3)
LENGTH_SAMPLES = 46  # spacing: 0.05 mm
SPAN_SAMPLES = 73  # spacing: 1/6 nm
REFINED_SEGMENTS = 1200  # winner is automatically rechecked with twice this count

# %% Refinement driver; the shared sweep settings are restored afterward
def run_refinement():
    if not COARSE_RESULTS.exists():
        raise FileNotFoundError('Run GratingSweep.py first to produce coarse results.')
    coarse = json.loads(COARSE_RESULTS.read_text())
    best = coarse.get('best_sampled_design')
    if best is None:
        raise ValueError('Coarse search has no eligible best design.')
    # Reject comparisons made with different physical or metric assumptions.
    expected = dict(center_wavelength_um=sweep.cmt.CENTER_WAVELENGTH_UM,
        narrow_width_um=sweep.cmt.NARROW_UM, wide_width_um=sweep.cmt.WIDE_UM,
        film_um=sweep.cmt.FILM_UM, etch_um=sweep.cmt.ETCH_UM,
        sidewall_deg=sweep.cmt.SIDEWALL_DEG,
        duty=sweep.cmt.DUTY, kappa_scale=sweep.cmt.KAPPA_SCALE,
        loss_db_per_cm=sweep.cmt.LOSS_DB_PER_CM,
        apodization_fraction=sweep.cmt.APODIZATION_FRACTION,
        concentration_window_ps=sweep.CONCENTRATION_WINDOW_PS,
        ranking_min_reflected_energy=sweep.MIN_REFLECTED_ENERGY_FOR_RANKING)
    for key, value in expected.items():
        if not np.isclose(coarse[key], value):
            raise ValueError(f'{key} changed since coarse sweep; rerun coarse search first.')
    lengths = np.linspace(*LENGTH_RANGE_MM, LENGTH_SAMPLES)
    spans = np.linspace(*SIGNED_SPAN_RANGE_NM, SPAN_SAMPLES)
    if lengths[0] <= 0 or LENGTH_SAMPLES < 3 or SPAN_SAMPLES < 3:
        raise ValueError('Choose a positive length range and >=3 samples per axis.')
    print(f'Refining {LENGTH_SAMPLES*SPAN_SAMPLES} designs: '
          f'L={lengths[0]:.3f}–{lengths[-1]:.3f} mm; '
          f'signed span={spans[0]:.3f}–{spans[-1]:.3f} nm.', flush=True)
    names = ['OUTPUT_DIR', 'LENGTHS_MM', 'SIGNED_PERIOD_SPANS_NM', 'SEGMENTS']
    saved = {name: getattr(sweep, name) for name in names}
    try:
        sweep.OUTPUT_DIR = REFINED_OUTPUT
        sweep.LENGTHS_MM = lengths
        sweep.SIGNED_PERIOD_SPANS_NM = spans
        sweep.SEGMENTS = REFINED_SEGMENTS
        table, info = sweep.run_sweep()
    finally:
        for name, value in saved.items():
            setattr(sweep, name, value)
    refined = info['best_sampled_design']
    if refined is None:
        raise RuntimeError('No eligible design in refined region; inspect metrics.csv.')
    on_edge = (np.isclose(refined['length_mm'], lengths[[0, -1]]).any()
               or np.isclose(refined['signed_period_span_nm'], spans[[0, -1]]).any())
    fine = info['best_design_2x_segments']
    comparison = dict(coarse_results=str(COARSE_RESULTS), coarse_best=best,
        refined_best=refined, refined_best_at_boundary=bool(on_edge),
        concentration_gain_percentage_points=100*(refined['pulse_concentration']-
                                                  best['pulse_concentration']),
        fwhm_change_ps=refined['fwhm_ps']-best['fwhm_ps'],
        convergence=dict(segments=REFINED_SEGMENTS, check_segments=2*REFINED_SEGMENTS,
            max_complex_reflection_difference=fine['max_complex_reflection_difference'],
            fwhm_change_ps=fine['fwhm_ps']-refined['fwhm_ps'],
            concentration_change=fine['pulse_concentration']-refined['pulse_concentration']),
        note='Finite local grid refinement. Inspect boundary flag, pulse pedestals, '
             'and model calibration before treating this as a physical design.')
    (REFINED_OUTPUT/'coarse_vs_refined.json').write_text(json.dumps(comparison, indent=2))
    # Preserve a shortlist for later validation, not just a single selected pixel.
    table[table.reflected_energy_fraction >= sweep.MIN_REFLECTED_ENERGY_FOR_RANKING].sort_values(
        'pulse_concentration', ascending=False).head(10).to_csv(
            REFINED_OUTPUT/'top_10_candidates.csv', index=False)
    print(json.dumps(comparison, indent=2))
    if on_edge:
        print('Best design touches refinement boundary: expand/shift the range before validation.')
    return table, comparison

# %% Run local refined sweep; no EME or API calls
if __name__ == '__main__':
    refined_metrics, refinement_summary = run_refinement()
