"""Replot saved P1.25 densities only: no inference, CFD, or array writes."""
from pathlib import Path
import hashlib
import json
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets/promo'
COLORS=('#bd741b','#087f8c','#23334d')
LABELS=('LF1 · 50 points','Refined prediction (HF-like)','HF reference · 800 points')
STYLES=('--','-',':')

def memory_gate():
    report=subprocess.check_output(['free','-m'],text=True)
    print(report)
    available=int(next(s for s in report.splitlines() if s.startswith('Mem:')).split()[-1])
    relative=next(s.split(':',2)[2] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
    directory=Path('/sys/fs/cgroup')/relative.lstrip('/')
    while directory.is_relative_to('/sys/fs/cgroup'):
        limit,usage=directory/'memory.max',directory/'memory.current'
        if limit.exists() and usage.exists() and limit.read_text().strip()!='max':
            available=min(available,(int(limit.read_text())-int(usage.read_text()))//2**20)
        if directory==Path('/sys/fs/cgroup'): break
        directory=directory.parent
    print(f'Effective available RAM: {available} MiB.')
    return available

def main():
    available=memory_gate()
    paths=[OUT/'arrays'/n for n in ('x.npy','sod_P1.25_lf1_density.npy','sod_P1.25_nnlci_density.npy','sod_P1.25_hf_density.npy')]
    x,*fields=[np.load(p) for p in paths]
    assert all(a.shape==(800,) and np.isfinite(a).all() for a in [x,*fields])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':14,'axes.titlesize':16,'text.color':'#23334d','axes.labelcolor':'#23334d','figure.facecolor':'white'})
    lo,hi=np.min(fields),np.max(fields); ylim=(lo-.06*(hi-lo),hi+.08*(hi-lo))
    def style(ax):
        ax.set(xlim=(0,1),ylim=ylim,xlabel='Position x')
        ax.set_xticks([0,.5,1]); ax.grid(alpha=.16); ax.spines[['top','right']].set_visible(False)
    fig,axes=plt.subplots(1,3,figsize=(12,5),dpi=160,sharex=True,sharey=True)
    titles=['LF1 baseline','Refined prediction\n(HF-like)','HF reference']
    for ax,values,color,ls,title in zip(axes,fields,COLORS,STYLES,titles):
        ax.plot(x,values,color=color,ls=ls,lw=2); style(ax); ax.set_title(title,pad=14)
    axes[0].set_ylabel('Density')
    fig.suptitle('Checkpoint result · Sod P1.25',fontsize=22,fontweight='bold',x=.055,ha='left',y=.97)
    fig.text(.055,.84,'LF1: 50 points  |  NNLCI inputs: LF1 + LF2 (100 points)  |  HF: 800 points',fontsize=14)
    fig.subplots_adjust(left=.08,right=.98,top=.68,bottom=.16,wspace=.16)
    for name in ['sod_P1.25_fidelity_lift_triptych.png','sod_P1.25_fidelity_lift_strip.png']:
        fig.savefig(OUT/name,facecolor='white')
    plt.close(fig)
    fig,(ax,err)=plt.subplots(2,1,figsize=(12,7.5),dpi=160,sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for values,color,ls,label in zip(fields,COLORS,STYLES,LABELS): ax.plot(x,values,color=color,ls=ls,label=label,lw=2)
    style(ax); ax.set(xlabel='',ylabel='Density'); ax.legend(frameon=False,fontsize=16)
    for values,color,ls in zip(fields[:2],COLORS[:2],STYLES[:2]): err.plot(x,values-fields[2],color=color,ls=ls,lw=2)
    extent=max(np.max(np.abs(np.asarray(fields[:2])-fields[2])),1e-12)*1.1
    style(err); err.set(ylim=(-extent,extent),ylabel='Density − HF')
    err.axhline(0,color=COLORS[2],lw=1)
    fig.suptitle('Checkpoint result · Sod P1.25',fontsize=22,fontweight='bold',x=.08,ha='left',y=.98)
    fig.subplots_adjust(top=.90,bottom=.10,left=.10,right=.98,hspace=.13)
    fig.savefig(OUT/'sod_P1.25_density_overlay.png',facecolor='white'); plt.close(fig)
    provenance=OUT/'provenance.json'; meta=json.loads(provenance.read_text())
    meta['presentation_replot']={
        'operation':'Layout-only replot of saved P1.25 arrays; no inference, numerical changes, or smoothing',
        'renderer':'scripts/replot_sod_promo.py','backend':'Agg','device':'CPU','versions':{'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'array_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'outputs':['sod_P1.25_fidelity_lift_triptych.png','sod_P1.25_fidelity_lift_strip.png','sod_P1.25_density_overlay.png'],
        'effective_available_memory_mib':available,'animation':'This replot generates no animation; permanent PNG strip retained'}
    provenance.write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__': main()
