# %% Guarded cloud EME validation of the local scalar coupled-mode approximation
"""Run cells in order. Default execution only builds/previews geometry locally.

Set REQUEST_ESTIMATE=True to test the API and upload jobs for estimates.
Paid solver execution additionally requires RUN_EME=True and passing the cap.
Estimate-only uploads do not launch the solver. This file does not configure keys.

Three uniform period samples and shared 20/40-repeat sweeps screen
coupling/detuning before testing chirped designs. EME includes multiple modes; CMT includes only forward/backward TE0.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web
import coupledmodemodel as cmt

# %% Controls: paid calculations OFF by default
VALIDATION_STAGE = 'representation'  # first: 'representation'; later: 'period_sweep'
REPRESENTATION_RUN_LABEL = 'targeted_TE0'  # preserve previous paid files
REPRESENTATION_PERIODS = 20
REPRESENTATION_MODES = 8
REPRESENTATION_TOLERANCE = 1e-3
REUSE_SAVED_RESULTS = True  # avoid rerunning an already-downloaded paid result
MAX_PORT_NEFF_ERROR = 0.05
MAX_PORT_NEFF_IMAG = 1e-3
REQUEST_ESTIMATE = True
RUN_EME = True
MAX_TOTAL_CREDITS = 1.0  # user-adjustable cap; never runs with unknown/nonfinite estimates
PROJECT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUTPUT_DIR = PROJECT_DIR / 'outputs/eme_validation/three_periods'
PERIOD_COUNTS = [20, 40]  # cheap propagation-only repetition sweep; shared modal solves
# Three physical periods: endpoints and center of the refined winner.
REFINED_RESULTS = PROJECT_DIR / 'outputs/grating_sweep/refined_length_chirp/configuration_and_best.json'
MODE_COUNTS = [8]  # convergence study: e.g. [8, 16, 24]
GRID_UM = 0.04  # repeat at 0.025 or finer to check transverse convergence
FREQUENCIES = np.sort(td.C_0/np.linspace(1.535, 1.595, 11))
DOMAIN_YZ_UM = (6.0, 5.0)
LEAD_UM = 0.8  # narrow-section leads between grating faces and EME ports
SIDEWALL_DEG = cmt.SIDEWALL_DEG
# Only ONE wide/narrow period is meshed; EME reuses its modes and overlaps.
# 11 frequencies are screening resolution, not a converged group-delay measurement.

# %% Builders (no API access)
def rib_structure(width, xmin, xmax, name):
    slab = cmt.FILM_UM-cmt.ETCH_UM
    bottom = width+2*cmt.ETCH_UM*np.tan(np.deg2rad(SIDEWALL_DEG))
    return td.Structure(name=name,
        geometry=td.PolySlab(axis=0, slab_bounds=(xmin, xmax), vertices=[
            (-bottom/2, slab), (bottom/2, slab),
            (width/2, cmt.FILM_UM), (-width/2, cmt.FILM_UM)]),
        medium=td.material_library['LiNbO3']['Zelmon1997'](1))


def center_period_um():
    indices, low, high = cmt.load_indices(cmt.MODE_CSV)
    f0 = td.C_0/cmt.CENTER_WAVELENGTH_UM
    if not low < f0 < high or min(FREQUENCIES) < low or max(FREQUENCIES) > high:
        raise ValueError('EME comparison frequencies exceed the local mode tables.')
    nbar = (1-cmt.DUTY)*float(indices[0](f0))+cmt.DUTY*float(indices[1](f0))
    return cmt.CENTER_WAVELENGTH_UM/(2*nbar)


def build_eme(periods, num_modes, period=None, explicit=False, repetition_sweep=True):
    period = center_period_um() if period is None else float(period)
    length = periods*period if explicit else period
    start, end = -length/2, length/2
    outer_start, outer_end = start-LEAD_UM, end+LEAD_UM
    slab = cmt.FILM_UM-cmt.ETCH_UM
    ln = td.material_library['LiNbO3']['Zelmon1997'](1)
    structures = [td.Structure(name='silica', medium=td.material_library['SiO2']['Palik_NoLoss'],
        geometry=td.Box(center=(0, 0, -5), size=(td.inf, td.inf, 10)))]
    if slab > 0:
        structures.append(td.Structure(name='LN_slab', medium=ln,
            geometry=td.Box(center=(0, 0, slab/2), size=(td.inf, td.inf, slab))))
    structures.append(rib_structure(cmt.NARROW_UM, -td.inf, td.inf, 'narrow_bus'))
    boundaries = [start]
    for j in range(periods if explicit else 1):
        x = start+j*period
        # WIDE section first; duty is the fraction of the period that is wide.
        structures.append(rib_structure(cmt.WIDE_UM, x, x+cmt.DUTY*period, f'wide_{j}'))
        boundaries.append(x+cmt.DUTY*period)
        boundaries.append(start+(j+1)*period)
    indices, _, _ = cmt.load_indices(cmt.MODE_CSV)
    f0 = td.C_0/cmt.CENTER_WAVELENGTH_UM
    target = float(indices[0](f0))
    spec = td.EMEModeSpec(num_modes=num_modes, num_pml=(12, 12),
        target_neff=target, increasing_mode_tolerance=1e-7,
        sort_spec=td.ModeSortSpec(filter_key='TE_fraction', filter_reference=.5,
                                 track_freq='central'))
    if explicit:
        grid = td.EMEExplicitGrid(boundaries=boundaries,
                                 mode_specs=[spec]*(len(boundaries)+1))
    else:
        grid = td.EMECompositeGrid(subgrid_boundaries=[start, end], subgrids=[
            td.EMEUniformGrid(num_cells=1, mode_spec=spec),
            td.EMEExplicitGrid(boundaries=[start+cmt.DUTY*period],
                mode_specs=[spec, spec], num_reps=periods, name='grating'),
            td.EMEUniformGrid(num_cells=1, mode_spec=spec)])
    return td.EMESimulation(
        center=(0, 0, cmt.FILM_UM/2), size=(outer_end-outer_start, *DOMAIN_YZ_UM),
        axis=0, freqs=FREQUENCIES, structures=structures, medium=td.Medium(permittivity=1),
        grid_spec=td.GridSpec.uniform(dl=GRID_UM),
        eme_grid_spec=grid,
        sweep_spec=(td.EMEPeriodicitySweep(num_reps=[{'grating': n} for n in PERIOD_COUNTS])
                    if repetition_sweep and not explicit else None),
        store_port_modes=True), period, periods*period

# %% Select by physical mode properties AND available S-matrix labels
# Missing mode 0 must never be replaced by the first available radiative mode.
def select_guided_port_modes(data):
    indices, _, _ = cmt.load_indices(cmt.MODE_CSV)
    ports = data.port_modes_tuple
    selections = []
    for side, port in enumerate(ports):
        dimension = 'mode_index_in' if side == 0 else 'mode_index_out'
        available = np.asarray(data.smatrix.S21.coords[dimension])
        frequency = np.asarray(port.n_complex.f)
        expected = indices[0](frequency)  # both leads have the narrow cross section
        rows = []
        eligible = []
        for label in np.asarray(port.n_complex.mode_index):
            n = np.asarray(port.n_complex.sel(mode_index=label))
            te = np.asarray(port.pol_fraction.te.sel(mode_index=label))
            error = float(np.max(abs(n.real-expected)))
            attenuation = float(np.max(abs(n.imag)))
            te_min = float(np.min(te))
            ok = (label in available and np.isfinite(n).all() and np.isfinite(te).all()
                  and te_min >= cmt.MIN_TE_FRACTION and error <= MAX_PORT_NEFF_ERROR
                  and attenuation <= MAX_PORT_NEFF_IMAG)
            rows.append(dict(mode_label=int(label), in_smatrix=bool(label in available),
                             min_TE_fraction=te_min, max_neff_error=error,
                             max_abs_neff_imag=attenuation, eligible=bool(ok)))
            if ok: eligible.append((error, int(label)))
        print(f'Port {side+1} mode diagnostics:\n'+pd.DataFrame(rows).to_string(index=False))
        if not eligible:
            raise ValueError('No guided TE0 candidate matches the local mode table in the '
                f'available S-matrix labels at port {side+1}. Saved result retained. '
                'Do not select the nearest label or rerun paid jobs blindly. Inspect mode '
                'fields and solver warnings. New builds now target the expected TE0 neff; '
                'a new estimate/run may be needed after reviewing the diagnostics.')
        selections.append(min(eligible)[1])
    return tuple(selections)


# %% Comparison with CMT: port reference planes de-embedded to grating faces
def compare_result(data, periods, num_modes, period, length, sweep_index=0):
    input_mode, output_mode = select_guided_port_modes(data)
    def coefficient(name):
        out_mode = input_mode if name == 'S11' else output_mode
        return getattr(data.smatrix, name).sel(mode_index_out=out_mode, mode_index_in=input_mode).isel(
            sweep_index=sweep_index, drop=True).sortby('f')
    s11, s21 = coefficient('S11'), coefficient('S21')
    frequency = np.asarray(s11.f)
    port1, port2 = data.port_modes_tuple
    te = np.asarray(port1.pol_fraction.te.sel(mode_index=input_mode))
    if np.any(te < cmt.MIN_TE_FRACTION):
        raise ValueError('EME port mode 0 is not consistently TE-like; inspect/reselect mode.')
    # Use EME's OWN propagation constants for removing lead phase.
    beta1 = 2*np.pi*frequency/cmt.c*np.asarray(
        port1.n_complex.sel(mode_index=input_mode).interp(f=frequency)).real
    beta2 = 2*np.pi*frequency/cmt.c*np.asarray(
        port2.n_complex.sel(mode_index=output_mode).interp(f=frequency)).real
    r_eme = np.asarray(s11)*np.exp(-2j*beta1*LEAD_UM*1e-6)
    t_eme = np.asarray(s21)*np.exp(-1j*(beta1+beta2)*LEAD_UM*1e-6)
    indices, _, _ = cmt.load_indices(cmt.MODE_CSV)
    r_cmt, t_cmt, _ = cmt.reflection_spectrum(frequency, *indices,
        length_um=length, period_center_um=period, period_span_um=0,
        apodization_fraction=0, segments=max(100, periods*4))
    _, delay_eme, _ = cmt.phase_derivatives(frequency, r_eme)
    _, delay_cmt, _ = cmt.phase_derivatives(frequency, r_cmt)
    # Absolute reflection phase can differ by constant grating-origin/modal gauge.
    # Compare delay plus phase variation, not an arbitrary constant phase offset.
    tag = f'P{period*1000:.4f}nm_N{periods}_M{num_modes}'
    comparison = pd.DataFrame(dict(frequency_hz=frequency, wavelength_um=cmt.c/frequency*1e6,
        r_eme_real=r_eme.real, r_eme_imag=r_eme.imag, r_cmt_real=r_cmt.real,
        r_cmt_imag=r_cmt.imag, R_eme=abs(r_eme)**2, R_cmt=abs(r_cmt)**2,
        T_TE0_eme=abs(t_eme)**2, T_cmt=abs(t_cmt)**2,
        delay_eme_ps=delay_eme, delay_cmt_ps=delay_cmt))
    comparison['period_nm'] = period*1000
    comparison['period_count'] = periods
    comparison['num_modes'] = num_modes
    comparison.to_csv(OUTPUT_DIR/f'{tag}_comparison.csv', index=False)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    wl = cmt.c/frequency*1e6
    for ax, a, b, label in zip(axes,
            [abs(r_eme)**2, abs(t_eme)**2, delay_eme],
            [abs(r_cmt)**2, abs(t_cmt)**2, delay_cmt],
            ['TE0 reflectance', 'TE0 transmittance', 'Reflection delay (ps)']):
        ax.plot(wl, a, label='EME'); ax.plot(wl, b, '--', label='Uncalibrated CMT')
        ax.set_ylabel(label); ax.legend()
    axes[-1].set_xlabel('Wavelength (µm)')
    fig.suptitle(f'Uniform grating: {periods} periods, {num_modes} EME modes')
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/f'{tag}_comparison.png', dpi=160)
    plt.show(); plt.close(fig)
    print('Compare band center/width and phase variation across MULTIPLE lengths. '
          'TE0 power missing from R+T may be other modes or radiation, not absorption.')
    return comparison

# %% Local preview, optional estimates, and guarded paid execution
def period_sweep_main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = []
    if REFINED_RESULTS.exists():
        refined = json.loads(REFINED_RESULTS.read_text())
        center = float(refined['center_period_um'])
        span = abs(refined['best_sampled_design']['signed_period_span_nm'])*1e-3
    else:
        center, span = center_period_um(), .008
    physical_periods = [center-span/2, center, center+span/2]
    for period in physical_periods:
        for modes in MODE_COUNTS:
            sim, period, _ = build_eme(PERIOD_COUNTS[0], modes, period)
            tag = f'P{period*1000:.4f}nm_M{modes}'
            sim.to_file(str(OUTPUT_DIR/f'{tag}_simulation.json'))
            cases.append((tag, sim, modes, period))
    (OUTPUT_DIR/'screening_configuration.json').write_text(json.dumps(dict(
        tidy3d_version=td.__version__, physical_periods_um=physical_periods,
        repetition_counts=PERIOD_COUNTS, mode_counts=MODE_COUNTS, grid_um=GRID_UM,
        frequencies_hz=FREQUENCIES.tolist(), domain_yz_um=DOMAIN_YZ_UM,
        lead_um=LEAD_UM, duty=cmt.DUTY, source_refinement=str(REFINED_RESULTS),
        note='Sparse screening, not converged validation of the full chirped grating. '
             'Modal reuse applies only to identical periods, not arbitrary chirp.'), indent=2))
    print('Period samples (nm):', [round(p*1000, 4) for p in physical_periods])
    print('Each job solves 4 physical cells (2 leads + 1 wide/narrow pair), '
          'then reuses them for repetition counts:', PERIOD_COUNTS)
    fig, ax = plt.subplots(figsize=(9, 3))
    cases[0][1].plot_eps(z=cmt.FILM_UM-.05, freq=td.C_0/cmt.CENTER_WAVELENGTH_UM, ax=ax)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/'geometry_preview.png', dpi=160)
    plt.show(); plt.close(fig)
    if not (REQUEST_ESTIMATE or RUN_EME):
        print('Local preview saved. Set REQUEST_ESTIMATE=True for API test and pricing. '
              'RUN_EME=False prevents paid execution.')
        return cases
    web.test()  # API connectivity only; no key configuration here
    jobs = []
    for tag, sim, modes, period in cases:
        job = web.Job(simulation=sim, task_name=f'LN_uniform_{tag}', folder_name='LN_EME_validation')
        estimated_cost = float(job.estimate_cost())  # upload/estimate BEFORE solver execution
        if not np.isfinite(estimated_cost) or estimated_cost < 0:
            raise RuntimeError('Missing or invalid cost estimate: refusing execution.')
        print(f'{tag}: estimated maximum {estimated_cost:.6f} FlexCredits')
        jobs.append((job, estimated_cost, tag, modes, period))
    total = sum(item[1] for item in jobs)
    print(f'TOTAL ESTIMATE: {total:.6f} FlexCredits; paid execution enabled: {RUN_EME}')
    (OUTPUT_DIR/'cost_estimates.json').write_text(json.dumps(
        dict(total_credits=total, jobs=[dict(name=j[2], credits=j[1], task_id=j[0].task_id)
                                      for j in jobs]), indent=2))
    if not RUN_EME:
        print('Estimate only. No solver jobs started. Set RUN_EME=True to execute.')
        return cases
    if not np.isfinite(MAX_TOTAL_CREDITS) or total > MAX_TOTAL_CREDITS:
        raise RuntimeError(f'Estimate {total:.6f} exceeds cap {MAX_TOTAL_CREDITS}; no jobs started.')
    comparisons = []
    for job, _, tag, modes, period in jobs:
        data = job.run(path=str(OUTPUT_DIR/f'{tag}_eme.hdf5'))
        for index, periods in enumerate(PERIOD_COUNTS):
            comparisons.append(compare_result(data, periods, modes, period,
                                               periods*period, sweep_index=index))
    full = pd.concat(comparisons, ignore_index=True)
    full.to_csv(OUTPUT_DIR/'all_comparisons.csv', index=False)
    summary = []
    for (period, count, modes), group in full.groupby(['period_nm', 'period_count', 'num_modes']):
        summary.append(dict(period_nm=period, period_count=int(count), num_modes=int(modes),
            reflectance_RMSE=float(np.sqrt(np.mean((group.R_eme-group.R_cmt)**2))),
            TE0_transmittance_RMSE=float(np.sqrt(np.mean((group.T_TE0_eme-group.T_cmt)**2)))))
    pd.DataFrame(summary).to_csv(OUTPUT_DIR/'comparison_errors.csv', index=False)
    fig, axes = plt.subplots(len(physical_periods), 2, figsize=(12, 3*len(physical_periods)),
                             squeeze=False)
    for row, period in enumerate(physical_periods):
        subset = full[np.isclose(full.period_nm, period*1000)]
        for (count, modes), group in subset.groupby(['period_count', 'num_modes']):
            label = f'N={count}, M={modes}'
            line, = axes[row, 0].plot(group.wavelength_um, group.R_eme, label=label+' EME')
            axes[row, 0].plot(group.wavelength_um, group.R_cmt, '--', color=line.get_color(),
                              label=label+' CMT')
            axes[row, 1].plot(group.wavelength_um, group.delay_eme_ps, color=line.get_color())
            axes[row, 1].plot(group.wavelength_um, group.delay_cmt_ps, '--', color=line.get_color())
        axes[row, 0].set_ylabel(f'Λ={period*1000:.2f} nm\nReflectance')
        axes[row, 1].set_ylabel('Delay (ps), screening only')
        axes[row, 0].legend(fontsize=7)
        for ax in axes[row]: ax.set_xlabel('Wavelength (µm)')
    fig.tight_layout(); fig.savefig(OUTPUT_DIR/'overview.png', dpi=170)
    plt.show(); plt.close(fig)
    return cases

# %% Explicit versus repeated-cell equivalence check, before period sweeps
# Both jobs represent the SAME uniform grating and use identical numerical settings.
def representation_check():
    destination = PROJECT_DIR / 'outputs/eme_validation/representation_check' / REPRESENTATION_RUN_LABEL
    destination.mkdir(parents=True, exist_ok=True)
    period = center_period_um()
    models = {}
    for name, explicit in [('repeated', False), ('explicit', True)]:
        sim, _, _ = build_eme(REPRESENTATION_PERIODS, REPRESENTATION_MODES,
                             period, explicit=explicit, repetition_sweep=False)
        sim.to_file(str(destination/f'{name}_simulation.json'))
        models[name] = sim
        print(f'{name}: geometry length {sim.size[0]:.4f} µm; '
              f'{sim.eme_grid.num_cells} physical EME cells; '
              f'effective grating length {REPRESENTATION_PERIODS*period:.4f} µm')
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for ax, (name, sim) in zip(axes, models.items()):
        sim.plot_eps(z=cmt.FILM_UM-.05, freq=td.C_0/cmt.CENTER_WAVELENGTH_UM, ax=ax)
        ax.set_title(name + (' (single-period template)' if name == 'repeated' else ' (all periods)'))
    fig.tight_layout(); fig.savefig(destination/'both_geometries.png', dpi=170)
    plt.show(); plt.close(fig)
    (destination/'configuration.json').write_text(json.dumps(dict(
        periods=REPRESENTATION_PERIODS, modes=REPRESENTATION_MODES,
        period_um=period, duty=cmt.DUTY, grid_um=GRID_UM,
        frequencies_hz=FREQUENCIES.tolist(), lead_um=LEAD_UM,
        tolerance=REPRESENTATION_TOLERANCE, tidy3d_version=td.__version__), indent=2))
    if not (REQUEST_ESTIMATE or RUN_EME):
        print('Representation check selected; period sweep NOT scheduled. '
              'Set REQUEST_ESTIMATE=True to estimate these TWO jobs.')
        return models
    web.test()
    jobs = {}
    prices = {}
    cached = {}
    for name in models:
        path = destination/f'{name}.hdf5'
        if REUSE_SAVED_RESULTS and path.exists():
            print(f'Reusing paid result: {path}')
            cached[name] = td.EMESimulationData.from_file(str(path))
            select_guided_port_modes(cached[name])
    for name, sim in models.items():
        if name in cached:
            prices[name] = 0.0
            continue
        job = web.Job(simulation=sim, task_name=f'LN_representation_{name}',
                      folder_name='LN_EME_validation')
        price = float(job.estimate_cost())
        if not np.isfinite(price) or price < 0:
            raise RuntimeError('Invalid estimate; refusing execution.')
        prices[name] = price
        jobs[name] = job
        print(f'{name}: estimated maximum {price:.6f} FlexCredits')
    total = sum(prices.values())
    print(f'TOTAL for both representations: {total:.6f} FlexCredits')
    (destination/'cost_estimates.json').write_text(json.dumps(
        dict(total_credits=total, jobs=prices), indent=2))
    if not RUN_EME:
        print('Estimate only. Neither solver job started.')
        return models
    if not np.isfinite(MAX_TOTAL_CREDITS) or total > MAX_TOTAL_CREDITS:
        raise RuntimeError('Combined estimate exceeds MAX_TOTAL_CREDITS; no jobs started.')
    spectra = {}
    for name in models:
        data = cached[name] if name in cached else jobs[name].run(path=str(destination/f'{name}.hdf5'))
        input_mode, output_mode = select_guided_port_modes(data)
        ports = data.port_modes_tuple
        columns = {}
        for label, component, port in [('r', 'S11', ports[0]), ('t', 'S21', ports[1])]:
            out_mode = input_mode if label == 'r' else output_mode
            result = getattr(data.smatrix, component).sel(mode_index_out=out_mode, mode_index_in=input_mode)
            if 'sweep_index' in result.dims:
                result = result.isel(sweep_index=0, drop=True)
            result = result.sortby('f')
            frequency = np.asarray(result.f)
            port_label = input_mode if label == 'r' else output_mode
            if np.any(np.asarray(port.pol_fraction.te.sel(mode_index=port_label)) < cmt.MIN_TE_FRACTION):
                raise ValueError('Port mode 0 is not consistently TE-like.')
            def beta(port_data, mode_label):
                n = port_data.n_complex.sel(mode_index=mode_label).interp(f=frequency)
                return 2*np.pi*frequency/cmt.c*np.asarray(n).real
            lead_phase = (2*beta(ports[0], input_mode) if label == 'r' else
                          beta(ports[0], input_mode)+beta(ports[1], output_mode))*LEAD_UM*1e-6
            columns[label] = np.asarray(result)*np.exp(-1j*lead_phase)
        spectra[name] = (frequency, columns)
    frequency, repeated = spectra['repeated']
    explicit_f, explicit = spectra['explicit']
    if not np.allclose(frequency, explicit_f, rtol=1e-12):
        raise ValueError('Result frequency axes disagree.')
    table = pd.DataFrame(dict(frequency_hz=frequency, wavelength_um=cmt.c/frequency*1e6))
    summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for row, label in enumerate(['r', 't']):
        a, b = repeated[label], explicit[label]
        # Report raw complex differences; additionally expose constant modal phase gauge.
        gauge = np.angle(np.vdot(a, b))
        summary[label] = dict(max_power_difference=float(np.max(abs(abs(a)**2-abs(b)**2))),
            max_complex_difference=float(np.max(abs(a-b))),
            constant_phase_offset_rad=float(gauge),
            max_complex_difference_after_constant_phase_alignment=float(
                np.max(abs(a*np.exp(1j*gauge)-b))))
        for name, values in [('repeated', a), ('explicit', b)]:
            table[f'{label}_{name}_real'] = values.real
            table[f'{label}_{name}_imag'] = values.imag
            axes[row, 0].plot(table.wavelength_um, abs(values)**2,
                              '--' if name == 'explicit' else '-', label=name)
        axes[row, 0].set_ylabel('Reflectance' if label == 'r' else 'Transmittance')
        axes[row, 0].legend()
        axes[row, 1].plot(table.wavelength_um, abs(a-b), label='Raw complex error')
        axes[row, 1].plot(table.wavelength_um, abs(a*np.exp(1j*gauge)-b),
                          '--', label='Constant-phase-aligned error')
        axes[row, 1].set_ylabel(f'|Δ{label}|'); axes[row, 1].legend(fontsize=8)
    summary['raw_complex_check_passed'] = all(
        summary[label]['max_complex_difference'] <= REPRESENTATION_TOLERANCE for label in ['r', 't'])
    summary['tolerance'] = REPRESENTATION_TOLERANCE
    summary['note'] = ('Representation equivalence only; not CMT or physical accuracy validation. '
                       'Spatial grid registration and modal phase gauges can affect comparison.')
    table.to_csv(destination/'representation_comparison.csv', index=False)
    (destination/'comparison_summary.json').write_text(json.dumps(summary, indent=2))
    for ax in axes[-1]: ax.set_xlabel('Wavelength (µm)')
    fig.tight_layout(); fig.savefig(destination/'representation_comparison.png', dpi=170)
    plt.show(); plt.close(fig)
    print(json.dumps(summary, indent=2))
    print('Period sweep has NOT been started. Inspect agreement before switching stages.')
    return models


def main():
    if VALIDATION_STAGE == 'representation':
        return representation_check()
    if VALIDATION_STAGE == 'period_sweep':
        return period_sweep_main()
    raise ValueError("VALIDATION_STAGE must be 'representation' or 'period_sweep'.")

# %% Run last: default = local previews of the two representations only
if __name__ == '__main__':
    validation_cases = main()

# %%
