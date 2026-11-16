# %% Offline EME audit and provisional CMT calibration -- NO CLOUD CALLS
"""Run all cells or execute this file. Reads existing HDF5/CSV files only.

Audit guided-channel power, port modes, solver warnings, phase references and
length dependence. Fit uniform two-wave CMT on three periods at N=20/80;
hold out the intervening periods AND all N=200 cases.

An effective attenuation fit is descriptive: missing power can be radiation,
mode-basis truncation, passive constraints, or other numerical error. This script
never overwrites the production grating/CMT settings or claims EME convergence.
"""
from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import tidy3d as td
import coupledmodemodel as cmt

# %% Paths and screening thresholds
PROJECT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
SOURCE_DIR = PROJECT_DIR / 'outputs/eme_validation/five_periods'
OUTPUT_DIR = PROJECT_DIR / 'outputs/eme_to_cmt_calibration'
MAX_GUIDED_IMAG_NEFF = 1e-5  # classify low-attenuation channels; not PML/radiation modes
MIN_TE_FRACTION = .9
PHASE_MIN_REFLECTANCE = .02
TRAIN_REPETITIONS = [20, 80]
# Global fits use one coupling scale and one constant detuning offset across all periods.
# They cannot represent arbitrary multimode or frequency-dependent radiation effects.

# %% Uniform CMT closed form; same convention as coupledmodemodel.py
def uniform_cmt(frequency, narrow_index, wide_index, period_um, length_um, duty,
                kappa_scale=1., detuning_offset_per_m=0., effective_loss_db_per_cm=0.):
    f = np.asarray(frequency)
    wavelength = cmt.c/f
    nn, nw = narrow_index(f), wide_index(f)
    kappa = kappa_scale*2*(nw-nn)*np.sin(np.pi*duty)/wavelength
    alpha = effective_loss_db_per_cm*100*np.log(10)/10
    delta = (2*np.pi*((1-duty)*nn+duty*nw)/wavelength-np.pi/(period_um*1e-6)
             + detuning_offset_per_m + .5j*alpha)
    length = length_um*1e-6
    q = np.sqrt(kappa*kappa-delta*delta+0j)
    s = length*np.sinc(1j*q*length/np.pi)
    denominator = np.cosh(q*length)-1j*delta*s
    r = 1j*kappa*s/denominator
    t = np.exp(1j*np.pi*length_um/period_um)/denominator
    return r, t


def interpolate_port(port, label, f):
    n = port.n_complex.sel(mode_index=label).interp(f=f)
    te = port.pol_fraction.te.sel(mode_index=label).interp(f=f)
    return np.asarray(n), np.asarray(te)

# %% Audit all saved EME results; never sum lossy/PML mode amplitudes as physical power
def audit_saved_results(config, indices):
    audit_rows, mode_rows, records, warning_rows = [], [], [], []
    periods = sorted(config['physical_periods_um'])
    for path in sorted(SOURCE_DIR.glob('*_eme.hdf5')):
        print(f'Auditing {path.name}', flush=True)
        data = td.EMESimulationData.from_file(str(path))
        # Physical template contains one period + leads, not the repeated effective length.
        period = float(data.simulation.size[0]-2*config['lead_um'])
        period_index = int(np.argmin(abs(np.asarray(periods)-period)))
        ports = data.port_modes_tuple
        f = np.asarray(data.smatrix.S21.f)
        expected = indices[0](f)
        log = str(data.log)
        (OUTPUT_DIR/f'{path.stem}_solver.log').write_text(log)
        warning_rows.append(dict(file=path.name, increasing_mode_warnings=log.count('Increasing EME modes'),
            discontinuity_warnings=log.count('discontinuous change'),
            constraint=data.simulation.constraint, diagnostics_available=data.diagnostics is not None,
            port_mode_frequency_samples=len(ports[0].n_complex.f), spectrum_samples=len(f)))
        labels, candidates = [], []
        for side, port in enumerate(ports):
            channel_labels = np.asarray(data.smatrix.S21.coords[
                'mode_index_in' if side == 0 else 'mode_index_out'])
            guided = []
            for label in np.asarray(port.n_complex.mode_index):
                n, te = interpolate_port(port, int(label), f)
                finite = np.isfinite(n).all() and np.isfinite(te).all()
                low_loss = finite and np.max(abs(n.imag)) < MAX_GUIDED_IMAG_NEFF
                available = label in channel_labels
                if low_loss and available: guided.append(int(label))
                for i, freq in enumerate(f):
                    mode_rows.append(dict(file=path.name, port=side+1, mode_label=int(label),
                        frequency_hz=freq, neff_real=n[i].real, neff_imag=n[i].imag,
                        TE_fraction=te[i], local_TE0_neff=expected[i],
                        neff_minus_local=n[i].real-expected[i],
                        available_in_smatrix=bool(available), low_attenuation=bool(low_loss)))
                if side == 0 and low_loss and available and np.min(te) >= MIN_TE_FRACTION:
                    candidates.append((float(np.max(abs(n.real-expected))), int(label)))
            labels.append(guided)
        if not candidates:
            raise ValueError(f'{path.name}: no low-attenuation TE input candidate; stop calibration.')
        _, input_label = min(candidates)
        # Select corresponding TE0 output independently; no assumption about label 0.
        output_candidates = []
        for label in labels[1]:
            n, te = interpolate_port(ports[1], label, f)
            if np.min(te) >= MIN_TE_FRACTION:
                output_candidates.append((float(np.max(abs(n.real-expected))), label))
        if not output_candidates: raise ValueError('No TE0 output candidate.')
        _, output_label = min(output_candidates)
        beta1 = 2*np.pi*f/cmt.c*interpolate_port(ports[0],input_label,f)[0].real
        beta2 = 2*np.pi*f/cmt.c*interpolate_port(ports[1],output_label,f)[0].real
        for sweep_index, repetitions in enumerate(config['repetition_counts']):
            s11 = data.smatrix.S11.isel(sweep_index=sweep_index).sel(mode_index_in=input_label)
            s21 = data.smatrix.S21.isel(sweep_index=sweep_index).sel(mode_index_in=input_label)
            r = np.asarray(s11.sel(mode_index_out=input_label))*np.exp(-2j*beta1*config['lead_um']*1e-6)
            t = np.asarray(s21.sel(mode_index_out=output_label))*np.exp(-1j*(beta1+beta2)*config['lead_um']*1e-6)
            # Low-attenuation normalized guided channels only. Radiation continuum and
            # nonorthogonal leaky-mode interference are not measured by this sum.
            Rguided = np.sum(abs(np.asarray(s11.sel(mode_index_out=labels[0])))**2, axis=-1)
            Tguided = np.sum(abs(np.asarray(s21.sel(mode_index_out=labels[1])))**2, axis=-1)
            train = period_index % 2 == 0 and repetitions in TRAIN_REPETITIONS
            record = dict(file=path.name, period_um=period, repetitions=repetitions,
                length_um=repetitions*period, f=f, r=r, t=t, train=train)
            records.append(record)
            for i, freq in enumerate(f):
                audit_rows.append(dict(file=path.name, period_nm=period*1000,
                    repetitions=repetitions, frequency_hz=freq, wavelength_um=cmt.c/freq*1e6,
                    input_mode=input_label, output_mode=output_label, train=train,
                    R_TE0=abs(r[i])**2, T_TE0=abs(t[i])**2,
                    R_guided_proxy=Rguided[i], T_guided_proxy=Tguided[i],
                    other_guided_power_proxy=Rguided[i]+Tguided[i]-abs(r[i])**2-abs(t[i])**2,
                    unaccounted_guided_power_proxy=1-Rguided[i]-Tguided[i]))
        del data
        gc.collect()
    if len(records) != len(periods)*len(config['repetition_counts']):
        raise ValueError('Saved cases do not cover the configured sweep.')
    modes, audit, warnings = pd.DataFrame(mode_rows),pd.DataFrame(audit_rows),pd.DataFrame(warning_rows)
    modes.to_csv(OUTPUT_DIR/'port_mode_audit.csv',index=False)
    audit.to_csv(OUTPUT_DIR/'power_audit.csv',index=False)
    warnings.to_csv(OUTPUT_DIR/'solver_warning_audit.csv',index=False)
    return records, audit, modes, warnings

# %% Fit on powers; phase is an independent diagnostic, not quietly fitted away
# Parameter units: [coupling scale, detuning / 1e4 m^-1, effective loss / 1000 dB/cm].
def fit_model(records, indices, duty, allow_effective_loss):
    training = [record for record in records if record['train']]
    def residual(parameters):
        chunks = []
        for record in training:
            r,t = predict(record,indices,duty,parameters)
            chunks.extend([abs(r)**2-abs(record['r'])**2,abs(t)**2-abs(record['t'])**2])
        return np.concatenate(chunks)
    lower = [.02,-20,0] if allow_effective_loss else [.02,-20]
    upper = [4,20,100] if allow_effective_loss else [4,20]
    fits=[]
    for k in [.3,.7,1.2]:
        for detuning in [-2,0,2]:
            initial=[k,detuning,1.] if allow_effective_loss else [k,detuning]
            fits.append(least_squares(residual,initial,bounds=(lower,upper),max_nfev=500,
                                      x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-10))
    best=min(fits,key=lambda result:np.sum(result.fun**2))
    return best


def predict(record,indices,duty,parameters):
    return uniform_cmt(record['f'],*indices,record['period_um'],record['length_um'],duty,
        kappa_scale=parameters[0],detuning_offset_per_m=parameters[1]*1e4,
        effective_loss_db_per_cm=parameters[2]*1000 if len(parameters)>2 else 0.)


def score_model(records,indices,duty,parameters,name):
    rows=[]
    for record in records:
        r,t=predict(record,indices,duty,parameters)
        mask=(abs(record['r'])**2>PHASE_MIN_REFLECTANCE)&(abs(r)**2>PHASE_MIN_REFLECTANCE)
        # Only a CONSTANT phase gauge is removed, never an adjustable delay or chirp.
        phase_error=np.nan
        if mask.sum()>=4:
            angle=np.angle(record['r'][mask]*r[mask].conj())
            gauge=np.angle(np.sum(np.exp(1j*angle)))
            phase_error=float(np.sqrt(np.mean(np.angle(np.exp(1j*(angle-gauge)))**2)))
        rows.append(dict(model=name,period_nm=record['period_um']*1000,
            repetitions=record['repetitions'],train=record['train'],
            reflectance_RMSE=float(np.sqrt(np.mean((abs(r)**2-abs(record['r'])**2)**2))),
            transmittance_RMSE=float(np.sqrt(np.mean((abs(t)**2-abs(record['t'])**2)**2))),
            reflection_phase_RMSE_rad_constant_gauge_removed=phase_error))
    return pd.DataFrame(rows)

# %% Full offline analysis and diagnostic plots
def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    config=json.loads((SOURCE_DIR/'screening_configuration.json').read_text())
    indices,_,_=cmt.load_indices(cmt.MODE_CSV)
    records,audit,modes,warnings=audit_saved_results(config,indices)
    # Verify optimized closed form against production CMT before fitting anything.
    probe=records[len(records)//2]
    r,t=predict(probe,indices,config['duty'],[1.,0.])
    rr,tt,_=cmt.reflection_spectrum(probe['f'],*indices,length_um=probe['length_um'],
        period_center_um=probe['period_um'],period_span_um=0,apodization_fraction=0,
        segments=200,duty=config['duty'],loss_db_per_cm=0,kappa_scale=1)
    assert np.allclose(r,rr,atol=1e-10) and np.allclose(t,tt,atol=1e-10)
    print('Fitting power spectra; held-out cases are never used in optimization.',flush=True)
    models={'uncalibrated':np.array([1.,0.])}
    fit_info={}
    for name,allow_loss in [('coupling_detuning',False),('effective_loss_diagnostic',True)]:
        fit=fit_model(records,indices,config['duty'],allow_loss)
        models[name]=fit.x
        fit_info[name]=dict(kappa_scale=float(fit.x[0]),detuning_offset_per_m=float(fit.x[1]*1e4),
            effective_loss_db_per_cm=float(fit.x[2]*1000) if allow_loss else 0.,
            optimizer_success=bool(fit.success),jacobian_rank=int(np.linalg.matrix_rank(fit.jac)),
            at_parameter_bound=bool(np.any(fit.active_mask)),training_objective=float(np.sum(fit.fun**2)))
    scores=pd.concat([score_model(records,indices,config['duty'],p,name) for name,p in models.items()])
    scores.to_csv(OUTPUT_DIR/'fit_errors_by_case.csv',index=False)
    summary=scores.groupby(['model','train'])[['reflectance_RMSE','transmittance_RMSE',
        'reflection_phase_RMSE_rad_constant_gauge_removed']].mean().reset_index()
    summary.to_csv(OUTPUT_DIR/'training_vs_heldout_errors.csv',index=False)
    # Full length/period coverage, not only a favorable fitted example.
    for n in config['repetition_counts']:
        cases=[record for record in records if record['repetitions']==n]
        fig,axes=plt.subplots(len(cases),2,figsize=(12,2.8*len(cases)),squeeze=False)
        for row,record in enumerate(cases):
            wl=cmt.c/record['f']*1e6
            for col,key in enumerate(['r','t']):
                axes[row,col].plot(wl,abs(record[key])**2,'k',lw=2,label='Saved EME TE0')
                for name,params in models.items():
                    r,t=predict(record,indices,config['duty'],params)
                    axes[row,col].plot(wl,abs(r if key=='r' else t)**2,'--',label=name)
                axes[row,col].set(xlabel='Wavelength (µm)',ylabel='Reflectance' if col==0 else 'Transmittance',
                    title=f'Λ={record["period_um"]*1000:.2f} nm, N={n}, '+('training' if record['train'] else 'HELD OUT'))
                axes[row,col].legend(fontsize=7)
        fig.tight_layout();fig.savefig(OUTPUT_DIR/f'calibration_N{n}.png',dpi=150);plt.close(fig)
    center=min(records,key=lambda rec:abs(rec['period_um']-config['physical_periods_um'][2]))['period_um']
    fig,axes=plt.subplots(2,1,figsize=(9,8),sharex=True)
    for n in config['repetition_counts']:
        subset=audit[np.isclose(audit.period_nm,center*1000)&(audit.repetitions==n)]
        axes[0].plot(subset.wavelength_um,subset.R_TE0+subset.T_TE0,label=f'TE0 sum, N={n}')
        axes[0].plot(subset.wavelength_um,subset.R_guided_proxy+subset.T_guided_proxy,'--',label=f'Guided sum proxy, N={n}')
        axes[1].plot(subset.wavelength_um,subset.other_guided_power_proxy,label=f'N={n}')
    axes[0].set_ylabel('Reflected + transmitted fraction');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Wavelength (µm)',ylabel='Other low-attenuation guided-channel power')
    axes[1].legend();fig.tight_layout();fig.savefig(OUTPUT_DIR/'power_balance_audit.png',dpi=160);plt.close(fig)
    # Audit analysis extraction/reference planes against already-exported comparison CSV.
    exported=pd.read_csv(SOURCE_DIR/'all_comparisons.csv')
    extraction_error=0.
    for record in records:
        previous=exported[np.isclose(exported.period_nm,record['period_um']*1000)&
                          (exported.period_count==record['repetitions'])].sort_values('frequency_hz')
        if len(previous)!=len(record['f']):
            raise ValueError('Existing comparison CSV does not match raw HDF5 frequency coverage.')
        original_r=previous.r_eme_real.to_numpy()+1j*previous.r_eme_imag.to_numpy()
        extraction_error=max(extraction_error,float(np.max(abs(record['r']-original_r))))
    # Phase is checked independently: no fitted delay is removed.
    case=min([record for record in records if record['repetitions']==80],
             key=lambda record:abs(record['period_um']-center))
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    frequencies=case['f'];wl=cmt.c/frequencies*1e6
    reference=np.unwrap(np.angle(case['r']))
    axes[0].plot(wl,reference-reference[len(reference)//2],'k',label='Saved EME')
    _,delay,_=cmt.phase_derivatives(frequencies,case['r'])
    axes[1].plot(wl,delay,'k',label='Saved EME')
    for name,params in models.items():
        r,_=predict(case,indices,config['duty'],params)
        phase=np.unwrap(np.angle(r))
        axes[0].plot(wl,phase-phase[len(phase)//2],'--',label=name)
        _,delay,_=cmt.phase_derivatives(frequencies,r)
        axes[1].plot(wl,delay,'--',label=name)
    axes[0].set_ylabel('Reflection phase relative to center (rad)')
    axes[1].set(xlabel='Wavelength (µm)',ylabel='Reflection delay (ps), masked at low R')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'phase_and_delay_audit.png',dpi=160);plt.close(fig)
    guided_te=modes[modes.low_attenuation & modes.available_in_smatrix &
                    (modes.TE_fraction>=MIN_TE_FRACTION)]
    fig,ax=plt.subplots(figsize=(8,4))
    for file,group in guided_te[guided_te.port==1].groupby('file'):
        ax.plot(cmt.c/group.frequency_hz*1e6,group.neff_minus_local,label=file.split('_M')[0])
    ax.set(xlabel='Wavelength (µm)',ylabel='EME TE0 neff − local-table TE0 neff')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUTPUT_DIR/'effective_index_mismatch.png',dpi=160);plt.close(fig)
    report=dict(status='PROVISIONAL DIAGNOSTIC CALIBRATION; not applied to compressor',
        source_dir=str(SOURCE_DIR),mode_csv=str(cmt.MODE_CSV),fits=fit_info,
        saved_csv_vs_raw_hdf5_max_complex_r_error=extraction_error,
        guided_TE0_neff_minus_local_range=[float(guided_te.neff_minus_local.min()),
                                         float(guided_te.neff_minus_local.max())],
        heldout_definition='All N=200 cases plus intervening period indices 1 and 3 at every length',
        warning_totals=dict(increasing_mode_warnings=int(warnings.increasing_mode_warnings.sum()),
                            discontinuity_warnings=int(warnings.discontinuity_warnings.sum())),
        max_other_guided_power_proxy=float(audit.other_guided_power_proxy.max()),
        unaccounted_guided_power_proxy_range=[float(audit.unaccounted_guided_power_proxy.min()),
                                            float(audit.unaccounted_guided_power_proxy.max())],
        diagnostics_stored=bool(warnings.diagnostics_available.all()),
        conclusions=['Repeated/explicit agreement checks representation equivalence only.',
            'The low-attenuation guided-channel sum is a diagnostic proxy, not a radiation power measurement.',
            'Numerical/passivity truncation and radiation cannot be separated from these outputs alone.',
            'Effective-loss fit is not a measured material/scattering loss.',
            'Fitting powers does not validate reflection phase, group delay or pulse compression.',
            'No additional paid jobs or local EME solves were run; no mesh/mode convergence is established.'])
    (OUTPUT_DIR/'audit_and_calibration.json').write_text(json.dumps(report,indent=2))
    print(summary.to_string(index=False));print(json.dumps(report,indent=2))
    return report,scores

# %% Execute offline only
if __name__=='__main__':
    calibration_report,calibration_scores=main()
