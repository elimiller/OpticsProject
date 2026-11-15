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