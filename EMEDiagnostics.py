# %% Small EME reference checks before further grating calibration
"""Local construction by default. No broadband mode interpolation.
Estimate all outstanding jobs before running any. Compatible cached results are
loaded without resubmission. This does not start the grating validation sweep.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web
import EMEvalidation as geometry

# %% Controls -- first run with estimate only
REQUEST_ESTIMATE = True
RUN_PAID = True
MAX_TOTAL_CREDITS = 1.0
# baseline: 3 jobs. convergence: insertion with more modes, finer mesh, wider domain.
# constraints: diagnostic comparison only; unitary is not proof of accuracy.
STAGE = 'convergence'  # 'baseline', 'convergence', 'constraints'
WAVELENGTHS_UM = np.array([1.55, 1.565, 1.58])
FREQUENCIES = np.sort(td.C_0/WAVELENGTHS_UM)
PROJECT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUTPUT_DIR = PROJECT_DIR/'outputs/eme_diagnostics'

# %% Build simple structures using exactly the grating material/axis conventions
def build(kind, modes=8, mesh=.04, domain=(6.,5.), constraint='passive'):
    period = geometry.center_period_um()
    width_length = geometry.cmt.DUTY*period
    template,_,_=geometry.build_eme(20,modes,period,repetition_sweep=False)
    structures = list(template.structures[:3])  # substrate, slab, narrow bus
    interfaces = []
    if kind == 'insertion':
        structures.append(geometry.rib_structure(geometry.cmt.WIDE_UM,
            -width_length/2,width_length/2,'single_wide_insertion'))
        interfaces=[-width_length/2,width_length/2]
    elif kind == 'straight_split':
        interfaces=[-width_length/2,width_length/2]  # same medium at all interfaces
    elif kind != 'straight':
        raise ValueError(kind)
    spec=td.EMEModeSpec(num_modes=modes,target_neff=template.eme_grid.mode_specs[0].target_neff,
        num_pml=(12,12),increasing_mode_tolerance=1e-7,interp_spec=None,
        sort_spec=td.ModeSortSpec(filter_key='TE_fraction',filter_reference=.5,track_freq='central'))
    return td.EMESimulation(center=(0,0,geometry.cmt.FILM_UM/2),
        size=(2*geometry.LEAD_UM+width_length,*domain),axis=0,freqs=FREQUENCIES,
        medium=template.medium,structures=structures,grid_spec=td.GridSpec.uniform(dl=mesh),
        eme_grid_spec=td.EMEExplicitGrid(boundaries=interfaces,
            mode_specs=[spec]*(len(interfaces)+1)),constraint=constraint,
        store_port_modes=True,eme_diagnostics=True)


def cases():
    if STAGE=='baseline':
        return {name:build(name) for name in ['straight','straight_split','insertion']}
    if STAGE=='convergence':
        return {'insertion':build('insertion'),
            'insertion_M16':build('insertion',modes=16),
            'insertion_M24':build('insertion',modes=24),
            'insertion_M16_mesh025':build('insertion',modes=16,mesh=.025),
            'insertion_M16_domain8x7':build('insertion',modes=16,domain=(8.,7.))}
    if STAGE=='constraints':
        return {'insertion':build('insertion'),
                'insertion_unconstrained':build('insertion',constraint=None),
                'insertion_unitary':build('insertion',constraint='unitary')}
    raise ValueError('Unknown STAGE')

# %% Analyze saved results using mode labels and physical TE0 selection
def analyze(name,data,destination):
    input_label,output_label=geometry.select_guided_port_modes(data)
    rows=[]
    f=np.asarray(data.smatrix.S21.f)
    for i,freq in enumerate(f):
        def column(component):
            a=getattr(data.smatrix,component).sel(f=freq,mode_index_in=input_label)
            if 'sweep_index' in a.dims:a=a.isel(sweep_index=0)
            return a
        r,t=column('S11'),column('S21')
        reflected=complex(r.sel(mode_index_out=input_label))
        transmitted=complex(t.sel(mode_index_out=output_label))
        guided=0.
        for port,a in zip(data.port_modes_tuple,[r,t]):
            for label in np.asarray(a.mode_index_out):
                n=complex(port.n_complex.sel(mode_index=int(label)).interp(f=freq))
                if np.isfinite(n) and abs(n.imag)<1e-5:
                    guided+=abs(complex(a.sel(mode_index_out=label)))**2
        rows.append(dict(case=name,wavelength_um=td.C_0/freq,
            R_TE0=abs(reflected)**2,T_TE0=abs(transmitted)**2,
            low_attenuation_guided_power_proxy=guided,
            unaccounted_power_proxy=1-guided,r_real=reflected.real,r_imag=reflected.imag,
            t_real=transmitted.real,t_imag=transmitted.imag))
    frame=pd.DataFrame(rows)
    frame.to_csv(destination/'metrics.csv',index=False)
    (destination/'solver.log').write_text(str(data.log))
    # Persist interface residuals in readable files where exposed by this version.
    diagnostic_manifest={}
    if data.diagnostics is not None:
        for key in type(data.diagnostics).model_fields:
            value=getattr(data.diagnostics,key)
            if hasattr(value,'to_dataframe'):
                try:
                    value.to_dataframe(name=key).reset_index().to_csv(destination/f'diagnostic_{key}.csv',index=False)
                    diagnostic_manifest[key]=f'diagnostic_{key}.csv'
                except (TypeError,ValueError):
                    diagnostic_manifest[key]='Stored in results.hdf5; inspect data.diagnostics.'
    (destination/'diagnostics_manifest.json').write_text(json.dumps(diagnostic_manifest,indent=2))
    modes=[]
    for side,port in enumerate(data.port_modes_tuple):
        for label in np.asarray(port.n_complex.mode_index):
            n=np.asarray(port.n_complex.sel(mode_index=label))
            te=np.asarray(port.pol_fraction.te.sel(mode_index=label))
            for j,freq in enumerate(np.asarray(port.n_complex.f)):
                modes.append(dict(port=side+1,mode_label=int(label),frequency_hz=freq,
                    neff_real=n[j].real,neff_imag=n[j].imag,TE_fraction=te[j]))
    pd.DataFrame(modes).to_csv(destination/'port_modes.csv',index=False)
    print(frame.to_string(index=False))
    return frame

# %% Cost gate and resume-safe execution
def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    prepared=[];completed=[]
    for name,sim in cases().items():
        # Content hash prevents stale paid results being silently reused after settings change.
        content=sim.model_dump_json()
        fingerprint=hashlib.sha256(content.encode()).hexdigest()[:12]
        destination=OUTPUT_DIR/f'{name}_{fingerprint}'
        destination.mkdir(exist_ok=True)
        sim.to_file(str(destination/'simulation.json'))
        print(name,':',sim.eme_grid.num_cells,'cells,',len(sim.freqs),
              'direct frequencies; interpolation =',sim.eme_grid.mode_specs[0].interp_spec)
        path=destination/'results.hdf5'
        if path.exists():
            completed.append(analyze(name,td.EMESimulationData.from_file(str(path)),destination))
        else:prepared.append((name,sim,destination))
    if prepared and (REQUEST_ESTIMATE or RUN_PAID):
        web.test()
        jobs=[]
        for name,sim,destination in prepared:
            job=web.Job(simulation=sim,task_name=f'LN_diagnostic_{name}',folder_name='LN_EME_diagnostics')
            cost=float(job.estimate_cost())
            if not np.isfinite(cost) or cost<0:raise RuntimeError('Invalid estimate; no execution.')
            jobs.append((name,job,cost,destination))
            print(f'{name}: {cost:.6f} FlexCredits')
        total=sum(item[2] for item in jobs)
        print(f'COMBINED estimate: {total:.6f} FlexCredits; RUN_PAID={RUN_PAID}')
        (OUTPUT_DIR/f'{STAGE}_estimates.json').write_text(json.dumps(dict(total_credits=total,
            jobs=[dict(name=name,credits=cost,task_id=job.task_id) for name,job,cost,_ in jobs]),indent=2))
        if RUN_PAID:
            if not np.isfinite(MAX_TOTAL_CREDITS) or total>MAX_TOTAL_CREDITS:
                raise RuntimeError('Total exceeds credit cap. No jobs started.')
            for name,job,_,destination in jobs:
                data=job.run(path=str(destination/'results.hdf5'))
                completed.append(analyze(name,data,destination))
    elif prepared:
        print('Local setup only. Set REQUEST_ESTIMATE=True to obtain pricing, RUN_PAID=False.')
    if completed:
        combined=pd.concat(completed,ignore_index=True)
        combined.to_csv(OUTPUT_DIR/f'{STAGE}_summary.csv',index=False)
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for name,g in combined.groupby('case'):
            for ax,column in zip(axes,['R_TE0','T_TE0','unaccounted_power_proxy']):
                ax.plot(g.wavelength_um,g[column],'o-',label=name)
                ax.set(xlabel='Wavelength (µm)',ylabel=column)
        axes[0].legend(fontsize=7);fig.tight_layout()
        fig.savefig(OUTPUT_DIR/f'{STAGE}_summary.png',dpi=170);plt.show();plt.close(fig)
    return prepared

# %% Execute -- default local preparation only
if __name__=='__main__':
    diagnostic_cases=main()

# %%
