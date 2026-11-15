# %% Local LN Bragg-grating cross-section lookup tables
"""Run cells in order, or run this file. No cloud jobs or credits are used.

Coordinates (um): propagation x, lateral y, vertical z. Film occupies
0 < z < film thickness. X-cut LN is mapped with its crystal X vertical,
crystal Z along lateral y (extraordinary axis), and crystal Y along x.
Widths refer to the TOP of the rib; sidewall angles are measured from vertical.
This is a local uniform-section approximation, not a periodic/Bloch calculation.
"""
from itertools import product
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.constants import c
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

# %% Configuration: start small, then enable fabrication-tolerance sweeps
PROJECT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUTPUT_DIR = PROJECT_DIR / 'outputs' / 'ln_bragg_modes'
# Change RUN_LABEL whenever changing mesh/material/configuration; avoids stale outputs.
RUN_LABEL = 'coarse_checking'
RUN_DIR = OUTPUT_DIR / RUN_LABEL
RUN_DIR.mkdir(parents=True, exist_ok=True)

WAVELENGTHS_UM = np.linspace(1.50, 1.62, 10)
TOP_WIDTHS_UM = [1.1, 1.5]
FILM_THICKNESSES_UM = [0.600]
ETCH_DEPTHS_UM = [0.300]  # optional tolerance sweep: [0.280, 0.300, 0.320]
SIDEWALL_ANGLES_DEG = [0.0]  # optional: [0.0, 10.0, 20.0]
NUM_MODES = 4
GRID_UM = 0.05  # repeat at finer mesh before trusting dispersion derivatives
DOMAIN_YZ_UM = (6.0, 5.0)  # also check convergence with a larger domain
SAVE_FIELDS = True  # HDF5 enables later field-overlap analysis; can consume disk space
RUN_SWEEP = True

# Parameters for a later coupled-mode / scattering-matrix model, not mode-solver axes.
NARROW_WIDTH_UM, WIDE_WIDTH_UM = 1.1, 1.5
WIDE_DUTY_CYCLES = [0.4, 0.5, 0.6]
PERIOD_CENTERS_UM = [0.406, 0.4105, 0.415]
PERIOD_SPANS_UM = [0.004, 0.008, 0.012]
GRATING_LENGTHS_UM = [1000.0, 1800.0, 2500.0]

LN = td.material_library['LiNbO3']['Zelmon1997'](1)
SILICA = td.material_library['SiO2']['Palik_NoLoss']
AIR = td.Medium(permittivity=1.0)

# %% Geometry and solver builders (no source or FDTD run needed)
def build_solver(width, film, etch, angle, wavelengths=WAVELENGTHS_UM):
    if not (width > 0 and 0 < etch <= film and 0 <= angle < 60):
        raise ValueError('Require positive width, 0 < etch <= film, and 0 <= angle < 60.')
    slab = film - etch
    # Wider bottom, fixed top width; slope is applied only to etched rib height.
    bottom_width = width + 2 * etch * np.tan(np.deg2rad(angle))
    structures = [td.Structure(
        geometry=td.Box(center=(0, 0, -5), size=(td.inf, td.inf, 10)),
        medium=SILICA, name='silica_substrate')]
    if slab > 0:
        structures.append(td.Structure(
            geometry=td.Box(center=(0, 0, slab / 2), size=(td.inf, td.inf, slab)),
            medium=LN, name='LN_slab'))
    structures.append(td.Structure(
        geometry=td.PolySlab(axis=0, slab_bounds=(-td.inf, td.inf), vertices=[
            (-bottom_width / 2, slab), (bottom_width / 2, slab),
            (width / 2, film), (-width / 2, film)]),
        medium=LN, name='LN_rib'))
    center = (0, 0, film / 2)
    sim = td.Simulation(
        center=center, size=(0, *DOMAIN_YZ_UM), medium=AIR,
        structures=structures, grid_spec=td.GridSpec.uniform(dl=GRID_UM),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()),
        run_time=1e-12)
    return ModeSolver(
        simulation=sim, plane=td.Box(center=center, size=(0, *DOMAIN_YZ_UM)),
        freqs=np.sort(td.C_0 / np.asarray(wavelengths)),
        mode_spec=td.ModeSpec(num_modes=NUM_MODES, num_pml=(12, 12),
            sort_spec=td.ModeSortSpec(filter_key='TE_fraction',
                                     filter_reference=0.5, track_freq='central')))


def mode_table(data, geometry):
    """Export every mode; TE-first ordering does not guarantee a bound TE0 mode."""
    rows = []
    for mode in range(NUM_MODES):
        n = data.n_complex.sel(mode_index=mode)
        freq = np.asarray(n.f)
        neff = np.asarray(n)
        # Smooth fit in scaled angular frequency. Derivatives are numerical estimates.
        omega = 2 * np.pi * freq
        omega0 = np.mean(omega)
        scale = np.ptp(omega) / 2
        u = (omega - omega0) / scale
        beta = omega / c * neff.real
        fit = Polynomial.fit(u, beta, deg=min(5, len(u) - 1)).convert()
        ng = c * fit.deriv(1)(u) / scale
        beta2 = fit.deriv(2)(u) / scale**2
        te = np.asarray(data.pol_fraction.te.sel(mode_index=mode))
        for i in range(len(freq)):
            rows.append(dict(**geometry, mode_index=mode,
                wavelength_um=td.C_0 / freq[i], frequency_hz=freq[i],
                neff_real=neff[i].real, neff_imag=neff[i].imag,
                TE_fraction=te[i], beta_rad_per_m=beta[i],
                group_index_fit=ng[i], beta2_ps2_per_m_fit=beta2[i] * 1e24,
                beta_fit_residual_rad_per_m=beta[i] - fit(u[i])))
    return pd.DataFrame(rows).sort_values(['mode_index', 'wavelength_um'])

# %% Preview nominal cross section before the sweep
preview_solver = build_solver(1.1, 0.6, 0.3, 0.0)
fig, ax = plt.subplots(figsize=(7, 5))
preview_solver.simulation.plot_eps(x=0, freq=td.C_0 / 1.55, ax=ax)
ax.set_title('LN rib on silica; air upper cladding (extraordinary axis: y)')
fig.savefig(RUN_DIR / 'cross_section.png', dpi=160, bbox_inches='tight')
plt.show()

# %% Local sweep and exports
if RUN_SWEEP:
    config = dict(tidy3d_version=td.__version__, wavelengths_um=WAVELENGTHS_UM.tolist(),
        widths_um=TOP_WIDTHS_UM, films_um=FILM_THICKNESSES_UM,
        etches_um=ETCH_DEPTHS_UM, sidewalls_deg=SIDEWALL_ANGLES_DEG,
        grid_um=GRID_UM, domain_yz_um=DOMAIN_YZ_UM, num_modes=NUM_MODES,
        LN_model='Zelmon1997; extraordinary axis y', cladding='air',
        substrate='SiO2 Palik_NoLoss; semi-infinite approximation',
        warning='Fit-derived dispersion requires mesh/domain/sampling convergence. '
                'Mode index is tracked across frequency, not across geometries.')
    (RUN_DIR / 'configuration.json').write_text(json.dumps(config, indent=2))
    tables = []
    geometries = list(product(TOP_WIDTHS_UM, FILM_THICKNESSES_UM,
                              ETCH_DEPTHS_UM, SIDEWALL_ANGLES_DEG))
    for index, (width, film, etch, angle) in enumerate(geometries):
        tag = f'w{width:.4f}_h{film:.4f}_e{etch:.4f}_a{angle:.2f}'
        print(f'Local solve {index + 1}/{len(geometries)}: {tag}', flush=True)
        solver = build_solver(width, film, etch, angle)
        data = solver.solve()  # LOCAL ONLY: no tidy3d.web imports or submissions
        if SAVE_FIELDS:
            data.to_file(str(RUN_DIR / f'{tag}_fields.hdf5'))
        geometry = dict(top_width_um=width, film_um=film, etch_um=etch,
                        sidewall_deg=angle)
        table = mode_table(data, geometry)
        table.to_csv(RUN_DIR / f'{tag}_modes.csv', index=False)
        tables.append(table)
        # Plot only center-frequency fields rather than hundreds of subplots.
        fig, ax = plt.subplots(figsize=(6, 4))
        np.abs(data.Ey.isel(f=len(data.n_complex.f)//2, mode_index=0)
               .squeeze(drop=True)).plot(x='y', y='z', ax=ax)
        ax.set_title(f'Candidate TE0 |Ey|: {tag}')
        fig.savefig(RUN_DIR / f'{tag}_Ey.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
    all_modes = pd.concat(tables, ignore_index=True)
    all_modes.to_csv(RUN_DIR / 'all_modes.csv', index=False)
    print(f'Saved lookup tables and fields to {RUN_DIR}')

# %% First-order Bragg design candidates: no reflection or compression prediction yet
# beta_narrow * (1-duty)*period + beta_wide * duty*period ~ pi.
# This phase-average approximation ignores interface phase and strong corrugation.
if RUN_SWEEP:
    candidates = []
    for film, etch, angle in product(FILM_THICKNESSES_UM, ETCH_DEPTHS_UM,
                                     SIDEWALL_ANGLES_DEG):
        selection = all_modes[(all_modes.mode_index == 0) &
            (all_modes.film_um == film) & (all_modes.etch_um == etch) &
            (all_modes.sidewall_deg == angle)]
        narrow = selection[np.isclose(selection.top_width_um, NARROW_WIDTH_UM)]
        wide = selection[np.isclose(selection.top_width_um, WIDE_WIDTH_UM)]
        if narrow.empty or wide.empty:
            print('Skipping period estimates: include both narrow and wide widths.')
            continue
        paired = narrow.merge(wide, on='wavelength_um', suffixes=('_n', '_w'))
        for duty in WIDE_DUTY_CYCLES:
            n_avg = (1-duty)*paired.neff_real_n + duty*paired.neff_real_w
            for wavelength, period in zip(paired.wavelength_um,
                                          paired.wavelength_um / (2*n_avg)):
                candidates.append(dict(film_um=film, etch_um=etch, sidewall_deg=angle,
                    wide_duty_cycle=duty, wavelength_um=wavelength,
                    bragg_period_um_estimate=period))
    pd.DataFrame(candidates).to_csv(RUN_DIR / 'bragg_period_estimates.csv', index=False)
    designs = [dict(period_center_um=p, period_span_um=s,
        period_start_um=p-s/2, period_end_um=p+s/2, length_um=length,
        wide_duty_cycle=duty, narrow_width_um=NARROW_WIDTH_UM,
        wide_width_um=WIDE_WIDTH_UM, orientation=orientation)
        for p, s, length, duty, orientation in product(PERIOD_CENTERS_UM,
            PERIOD_SPANS_UM, GRATING_LENGTHS_UM, WIDE_DUTY_CYCLES, [1, -1])]
    pd.DataFrame(designs).to_csv(RUN_DIR / 'grating_design_sweep.csv', index=False)
    # Orientation -1 means reversing period_start/end in the propagation direction.
    print('Design sweep saved. Next: calibrate coupling/interface scattering, then '
          'compute complex reflection and apply it to the incoming chirped pulse.')

# %%
