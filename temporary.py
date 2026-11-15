# %% 
# %% imports
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.web import run as run_mode_solver
import tidy3d.web as web
import pandas as pd

import sys
import os
import json

# import custom wrapper functions
_path = os.getcwd()
folder = os.path.dirname(os.path.dirname(_path))

function_path = folder + "/Functions/"
sys.path.append(function_path)

# from tidy3D_UtilityFunctions import *
# from IO_UtilityFunctions import *

###################################################################

# %% directory management

# layer config path
layer_config_path = folder + "/Example Files/"

# input path if want to use a generated medium; otherwise None
medium_file = "/home/hlee/Documents/GDS/tidy3D/20260722/PN_medium.json"

###################################################################

# %% save path


layer_choice = "PN_493"

data_path = "/home/hlee/Documents/GDS/tidy3D/2026_DrFluffykins_TAPEOUT/dbr wdm/"

save_folder = "PN LUT/"
save_path = CheckMakeDirectory(data_path, save_folder)
save_name = "PN_dbr_LUT"

###################################################################

# %% input material parameters

with open(layer_config_path + "layer_config.json", "r") as f:
    layer_config_dict = json.load(f)

config_params = layer_config_dict[layer_choice]
nDevice = config_params["nDevice"]
wvg_thick = config_params["layer_thickness"]
box_thick = config_params["box_thick"]
sidewall_angle = config_params["sidewall_angle"]

print("Layer initialized: ", layer_choice)
print("refractive index: ", nDevice)
print("wvg thickness: ", wvg_thick)
print("box_thick: ", box_thick)
print("sidewall_angle: ", sidewall_angle)

# material definitions
mat_si = td.material_library["cSi"]["Palik_Lossless"]
mat_sio2 = td.material_library["SiO2"]["Palik_Lossless"]
mat_air = td.Medium(permittivity=1, name="air")  # type: ignore

mat_device = mat_si
mat_cladding = mat_sio2

if medium_file is None:
    print("Default material")
else:
    mat_device = td.PoleResidue.from_file(medium_file)

###################################################################

# %% input parameters

# Define wavelength/frequency ranges
target_wls = np.arange(0.3, 0.75 + 0.001, 0.02)
print(f"Solving for {len(target_wls)} wavelengths")
print(target_wls)

# target_wls = np.asarray([0.4, 1.0])

target_bw = 0.01  # for solver for each target wl
num_wls = 5

# Define geometries and their structures
# wvg_width_list = np.arange(0.05, 1.5 + 0.001, 0.05)
wvg_width_list = np.arange(0.7, 1.5 + 0.001, 0.05)
print(f"Solving for {len(wvg_width_list)} widths")
print(wvg_width_list)

# wvg_width_list = np.asarray([0.5, 1.0])

# run time and resolution arguments
run_time = 1e-12
min_steps_per_wvl = 18

# mode solver parameters
num_modes = 6
mode_filter_key = "TE_fraction"
mode_filter_reference = 0.5
track_freq = "central"

# control bool
remote = True

###################################################################

# %% initialize params
# intialize some params

eps_max = np.real(mat_device.eps_model(td.C_0 / min(target_wls)))

size_x = max(target_wls)
size_y = max([max(target_wls) * 4, max(wvg_width_list) * 3])
size_z = max([max(target_wls) * 4, 1.5 * box_thick])

print(size_y, size_z)

center_x = 0
center_y = 0
center_z = wvg_thick / 2

eff_inf = 1000

monitor_width = eff_inf
monitor_height = eff_inf

target_freqs = td.C_0 / target_wls
fwidth = np.abs(td.C_0 / min(target_wls) - td.C_0 / max(target_wls))

center_coord = [center_x, center_y, center_z]
monitor_size = [0, monitor_height, monitor_width]

###################################################################

# %% loop through all widths

df_array = []

for idx, wvg_width in enumerate(wvg_width_list):
    print(f"Solving for wvg width of: {wvg_width}")

    ### define structures ###
    dx = wvg_thick * np.tan(sidewall_angle * np.pi / 180)

    wvg_vertices = [
        [-wvg_width / 2, 0],
        [wvg_width / 2, 0],
        [wvg_width / 2 - dx, wvg_thick],
        [-wvg_width / 2 + dx, wvg_thick],
    ]
    wvg_geo = td.PolySlab(
        axis=0,
        slab_bounds=[-eff_inf, eff_inf],
        vertices=wvg_vertices,
    )  # type:ignore

    waveguide = td.Structure(
        geometry=wvg_geo,
        medium=mat_device,
        name="waveguide",
    )  # type:ignore

    # add a mode source
    src = td.ModeSource(
        center=center_coord,
        size=monitor_size,
        source_time=td.GaussianPulse(
            freq0=np.mean(target_freqs),
            fwidth=fwidth * 2,
        ),  # type: ignore
        direction="+",
        mode_index=0,
        name="mode_input",
    )  # type: ignore

    # create sim
    sim_3D = td.Simulation(
        size=[size_x, size_y, size_z],
        center=center_coord,
        structures=[waveguide],
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=min_steps_per_wvl, wavelength=min(target_wls)
        ),
        sources=[src],
        monitors=[],
        run_time=run_time,
        medium=mat_cladding,
        symmetry=(0, 0, 0),  # add symmetry to reduce cost
    )  # type:ignore

    if idx == 0:
        # show mesh grid; verify parameters
        ax = plot_sim_grid(
            sim=sim_3D,
            xlim_list=[-wvg_width, wvg_width],
            ylim_list=[-wvg_thick / 2, 2 * wvg_thick],
            x_axis=True,
            y_axis=False,
            z_axis=False,
        )
        plt.show()

        # print grid sizing
        print(
            f"Grid size x: {round(np.min(sim_3D.grid.sizes.x), 6), round(np.max(sim_3D.grid.sizes.x), 6)}"
        )
        print(
            f"Grid size y: {round(np.min(sim_3D.grid.sizes.y), 6), round(np.max(sim_3D.grid.sizes.y), 6)}"
        )
        print(
            f"Grid size z: {round(np.min(sim_3D.grid.sizes.z), 6), round(np.max(sim_3D.grid.sizes.z), 6)}"
        )

        # push to server to check costs/ check placements
        init_job = web.Job(
            simulation=sim_3D,
            task_name=save_name + "_visualize",
            folder_name=save_name,
            verbose=False,
        )  # type: ignore
        estimated_cost = float(web.estimate_cost(init_job.task_id))  # type: ignore

        # visualize
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, tight_layout=True, figsize=(15, 8))
        sim_3D.plot_eps(z=0, ax=ax1, eps_lim=(1, eps_max))
        sim_3D.plot_eps(y=0, ax=ax2, eps_lim=(1, eps_max))
        sim_3D.plot_eps(x=0, ax=ax3, eps_lim=(1, eps_max))
        plt.show()

    ### solve for modes
    solver_center = center_coord
    solver_size = monitor_size

    # create a mode sort spec
    sort_spec = td.ModeSortSpec(
        filter_key=mode_filter_key,
        filter_reference=mode_filter_reference,
        track_freq=track_freq,
    )  # type:ignore
    mode_spec = td.ModeSpec(
        num_modes=num_modes,
        sort_spec=sort_spec,
    )  # type:ignore

    # create mode solver object and then solve
    mode_input_plane = td.Box(
        center=solver_center,
        size=solver_size,
    )  # type:ignore

    mode_solver = ModeSolver(
        simulation=sim_3D,
        plane=mode_input_plane,
        freqs=target_freqs,
        mode_spec=mode_spec,
    )  # type:ignore

    if not remote:
        print("Solving modes locally...")
        mode_data = mode_solver.solve()
    else:
        print("Solving modes remotely...")
        mode_data = run_mode_solver(
            mode_solver,
            task_name=save_name + f"_{wvg_width}width",
            reduce_simulation=True,
            results_file=save_path + save_name + f"_{wvg_width}width" + "modes.hdf5",
        )  # type: ignore

    # plot all data for a given wvg width, for Ex
    fig, ax = plt.subplots(
        mode_spec.num_modes,
        len(target_freqs),
        tight_layout=True,
        figsize=(len(target_freqs), mode_spec.num_modes),
    )
    for j in range(mode_spec.num_modes):
        for i, freq in enumerate(target_freqs):
            Ex_data = mode_data.Ex.isel(mode_index=j, f=i, drop=True).squeeze().real.T
            ax[j, i].imshow(
                Ex_data,
                cmap="RdBu",
                origin="lower",
                vmax=max(np.abs(Ex_data.values).flatten()),
                vmin=-max(np.abs(Ex_data.values).flatten()),
            )

            if j == 0:
                ax[0, i].set_title(f"{round(td.C_0 / freq, 3)} um")

    plt.suptitle(f"Ex for {wvg_width} um Waveguide Width")
    plt.tight_layout()
    plt.savefig(save_path + save_name + f"_{wvg_width}width_Ex.png")
    plt.show()

    # plot all data for a given wvg width, for Ey
    fig, ax = plt.subplots(
        mode_spec.num_modes,
        len(target_freqs),
        tight_layout=True,
        figsize=(len(target_freqs), mode_spec.num_modes),
    )
    for j in range(mode_spec.num_modes):
        for i, freq in enumerate(target_freqs):
            Ey_data = mode_data.Ey.isel(mode_index=j, f=i, drop=True).squeeze().real.T
            ax[j, i].imshow(
                Ey_data,
                cmap="RdBu",
                origin="lower",
                vmax=max(np.abs(Ey_data.values).flatten()),
                vmin=-max(np.abs(Ey_data.values).flatten()),
            )

            if j == 0:
                ax[0, i].set_title(f"{round(td.C_0 / freq, 3)} um")

    plt.suptitle(f"Ey for {wvg_width} um Waveguide Width")
    plt.tight_layout()
    plt.savefig(save_path + save_name + f"_{wvg_width}width_Ey.png")
    plt.show()

    # plot all data for a given wvg width, for Ez
    fig, ax = plt.subplots(
        mode_spec.num_modes,
        len(target_freqs),
        tight_layout=True,
        figsize=(len(target_freqs), mode_spec.num_modes),
    )
    for j in range(mode_spec.num_modes):
        for i, freq in enumerate(target_freqs):
            Ez_data = mode_data.Ez.isel(mode_index=j, f=i, drop=True).squeeze().real.T
            ax[j, i].imshow(
                Ez_data,
                cmap="RdBu",
                origin="lower",
                vmax=max(np.abs(Ez_data.values).flatten()),
                vmin=-max(np.abs(Ez_data.values).flatten()),
            )

            if j == 0:
                ax[0, i].set_title(f"{round(td.C_0 / freq, 3)} um")

    plt.suptitle(f"Ez for {wvg_width} um Waveguide Width")
    plt.tight_layout()
    plt.savefig(save_path + save_name + f"_{wvg_width}width_Ez.png")
    plt.show()

    # convert data to dataframe and then save as csv
    mode_data_df = mode_data.to_dataframe()

    # append the wvg width as an extra column
    mode_data_df["wvg_width (um)"] = [
        wvg_width for _ii in range(len(target_wls) * num_modes)
    ]

    # append df to array
    df_array.append(mode_data_df)

    # save to csv
    mode_data_df.to_csv(save_path + save_name + f"_{wvg_width}width_data.csv")

###################################################################

# %% now merge all the data frames

full_df = pd.concat(df_array)
# save to csv
full_df.to_csv(save_path + save_name + "_data.csv")

# %%
##EME Validatoin
# %% Guarded cloud EME validation of the local scalar coupled-mode approximation
"""Run cells in order. Default execution only builds/previews geometry locally.

Set REQUEST_ESTIMATE=True to test the API and upload jobs for estimates.
Paid solver execution additionally requires RUN_EME=True and passing the cap.
Estimate-only uploads do not launch the solver. This file does not configure keys.

Uniform finite gratings are intentional: calibrate coupling/detuning before testing
chirped designs. EME includes multiple modes; CMT includes only forward/backward TE0.
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
REQUEST_ESTIMATE = True
RUN_EME = False
MAX_TOTAL_CREDITS = 1.0  # user-adjustable cap; never runs with unknown/nonfinite estimates
PROJECT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUTPUT_DIR = PROJECT_DIR / 'outputs/eme_validation/uniform'
PERIOD_COUNTS = [20]  # then compare e.g. [20, 40, 80] after checking prices
MODE_COUNTS = [8]  # convergence study: e.g. [8, 16, 24]
GRID_UM = 0.04  # repeat at 0.025 or finer to check transverse convergence
FREQUENCIES = np.sort(td.C_0/np.linspace(1.545, 1.585, 31))
DOMAIN_YZ_UM = (6.0, 5.0)
LEAD_UM = 0.8  # narrow-section leads between grating faces and EME ports
SIDEWALL_DEG = cmt.SIDEWALL_DEG
# Explicit longitudinal cells align exactly with every wide/narrow interface.
# No repeated-cell optimization yet: start with the SHORT uniform test above.

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


def build_eme(periods, num_modes):
    period = center_period_um()
    length = periods*period
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
    for j in range(periods):
        x = start+j*period
        # WIDE section first; duty is the fraction of the period that is wide.
        structures.append(rib_structure(cmt.WIDE_UM, x, x+cmt.DUTY*period, f'wide_{j}'))
        boundaries.append(x+cmt.DUTY*period)
        boundaries.append(start+(j+1)*period)
    spec = td.EMEModeSpec(num_modes=num_modes, num_pml=(12, 12),
        sort_spec=td.ModeSortSpec(filter_key='TE_fraction', filter_reference=.5,
                                 track_freq='central'))
    return td.EMESimulation(
        center=(0, 0, cmt.FILM_UM/2), size=(outer_end-outer_start, *DOMAIN_YZ_UM),
        axis=0, freqs=FREQUENCIES, structures=structures, medium=td.Medium(permittivity=1),
        grid_spec=td.GridSpec.uniform(dl=GRID_UM),
        eme_grid_spec=td.EMEExplicitGrid(boundaries=boundaries,
                                       mode_specs=[spec]*(len(boundaries)+1)),
        store_port_modes=True), period, length

# %% Comparison with CMT: port reference planes de-embedded to grating faces
def compare_result(data, periods, num_modes, period, length):
    def coefficient(name):
        return getattr(data.smatrix, name).sel(mode_index_out=0, mode_index_in=0).isel(
            sweep_index=0, drop=True).sortby('f')
    s11, s21 = coefficient('S11'), coefficient('S21')
    frequency = np.asarray(s11.f)
    port1, port2 = data.port_modes_tuple
    te = np.asarray(port1.pol_fraction.te.sel(mode_index=0))
    if np.any(te < cmt.MIN_TE_FRACTION):
        raise ValueError('EME port mode 0 is not consistently TE-like; inspect/reselect mode.')
    # Use EME's OWN propagation constants for removing lead phase.
    beta1 = 2*np.pi*frequency/cmt.c*np.asarray(
        port1.n_complex.sel(mode_index=0).interp(f=frequency)).real
    beta2 = 2*np.pi*frequency/cmt.c*np.asarray(
        port2.n_complex.sel(mode_index=0).interp(f=frequency)).real
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
    tag = f'N{periods}_M{num_modes}'
    pd.DataFrame(dict(frequency_hz=frequency, wavelength_um=cmt.c/frequency*1e6,
        r_eme_real=r_eme.real, r_eme_imag=r_eme.imag, r_cmt_real=r_cmt.real,
        r_cmt_imag=r_cmt.imag, R_eme=abs(r_eme)**2, R_cmt=abs(r_cmt)**2,
        T_TE0_eme=abs(t_eme)**2, T_cmt=abs(t_cmt)**2,
        delay_eme_ps=delay_eme, delay_cmt_ps=delay_cmt)).to_csv(
            OUTPUT_DIR/f'{tag}_comparison.csv', index=False)
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

# %% Local preview, optional estimates, and guarded paid execution
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = []
    for periods in PERIOD_COUNTS:
        for modes in MODE_COUNTS:
            sim, period, length = build_eme(periods, modes)
            tag = f'N{periods}_M{modes}'
            sim.to_file(str(OUTPUT_DIR/f'{tag}_simulation.json'))
            cases.append((tag, sim, periods, modes, period, length))
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
    for tag, sim, periods, modes, period, length in cases:
        job = web.Job(simulation=sim, task_name=f'LN_uniform_{tag}', folder_name='LN_EME_validation')
        estimated_cost = float(job.estimate_cost())  # upload/estimate BEFORE solver execution
        if not np.isfinite(estimated_cost) or estimated_cost < 0:
            raise RuntimeError('Missing or invalid cost estimate: refusing execution.')
        print(f'{tag}: estimated maximum {estimated_cost:.6f} FlexCredits')
        jobs.append((job, estimated_cost, tag, periods, modes, period, length))
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
    for job, _, tag, periods, modes, period, length in jobs:
        data = job.run(path=str(OUTPUT_DIR/f'{tag}_eme.hdf5'))
        compare_result(data, periods, modes, period, length)
    return cases

# %% Run this cell last; default is LOCAL PREVIEW ONLY
if __name__ == '__main__':
    validation_cases = main()

# %%
web.test()
# %%
