"""Stationary NYC startup-delay experiment. See protocol.txt for the frozen protocol."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/nyc-delay-mpl')
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, xlogy
from scipy.optimize import minimize
import solvers

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'build'   # processed data written by build/build_data.py and build/make_cache.py
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
NAMES = ['passenger_count','distance_mi','pickup_in_manhattan','dropoff_in_manhattan',
         'pickup_near_airport','dropoff_near_airport','traffic_congestion',
         'pickup_in_business_district','dropoff_in_business_district',
         'is_morning_commute','is_evening_commute','is_biz_to_airport','is_airport_to_biz',
         'had_rain','had_snow_fall','is_extreme_temp','is_long_daylight','is_weekend','is_night','is_cross_borough']
SHIFTS = {2:.05,3:.05,4:.20,5:.20,6:.10,7:.10,8:.10,9:.10,10:.10,
          13:.10,14:.15,15:.10,16:-.05,17:-.05,18:.15,19:.05}
DAY = 86_400_000_000_000


def design(z, price):
    return np.column_stack((np.ones(len(z)), z, z*(price/10)[:,None]))


def daylight(t):
    w = pd.read_csv(SOURCE.parent/'Simulator/daily_weather_data_2009_2015.csv')
    key = pd.to_datetime(w.date).to_numpy('datetime64[D]').astype('int64')
    values = pd.Series((w.daylight_duration.to_numpy() >= 12*3600).astype(float), index=key)
    out = values.reindex(np.asarray(t,dtype=np.int64)//DAY).to_numpy()
    assert np.isfinite(out).all()
    return out


def median_multiplier(x, base):
    out = np.full(len(x), base)
    for j, shift in SHIFTS.items():
        out += shift*x[:,j]
    out += np.clip(-.03*(x[:,1]-3), -.30, .10)
    out += .05*np.clip(x[:,0]-1, 0, 3)
    return np.maximum(out,1.1)


def fit_logit(blocks):
    theta = np.zeros(41)
    for it in range(40):
        g = np.zeros(41); h = 1e-8*np.eye(41); loss=0
        for phi,y in blocks:
            for start in range(0,len(y),100000):
                p = phi[start:start+100000].astype(float)
                yy = y[start:start+100000]; u=p@theta; mu=expit(u)
                loss += np.sum(np.logaddexp(0,u)-yy*u)
                g += p.T@(mu-yy)
                p *= np.sqrt(mu*(1-mu))[:,None]
                h += p.T@p
        step=np.linalg.solve(h,g); theta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    else:
        raise RuntimeError('Simulator Newton fit failed to converge')
    return theta,dict(iterations=it+1,nll=float(loss),max_score=float(np.max(np.abs(g))))


def prepare():
    DATA.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    df=pd.read_parquet(SOURCE/'data/processed/fit.parquet')
    excluded=df.time.between('2014-11-17','2014-11-24',inclusive='left')
    df=df[~excluded].sample(2_000_000,random_state=2025).sort_values('time')
    old=NAMES.copy();old[16]='is_long_sunshine'
    x=df[old].to_numpy(float); x[:,16]=daylight(df.time.to_numpy('datetime64[ns]').astype('int64'))
    scale=np.ones(20);scale[:2]=x[:,:2].std(axis=0)
    np.save(DATA/'scale.npy',scale);z=x/scale;fare=df.fare.to_numpy(float)
    meta=dict(fit_trips=len(df),offers_per_trip=2,excluded_validation_rows=int(excluded.sum()),
              fit_min=str(df.time.min()),fit_max=str(df.time.max()),features=NAMES,scale=scale.tolist(),
              sigma=.3,offer_distribution='log-uniform [0.4,4] times fare, clipped to [2.5,70]',
              daylight_definition='daylight_duration >= 12 hours',seed=2025,scenarios={})
    for base in (1.4,1.6,1.8):
        name=f'wtp{round(base*10)}';rng=np.random.default_rng(2025);t0=time.perf_counter()
        wtp=fare*median_multiplier(x,base)*np.exp(.3*rng.standard_normal(len(x)))
        blocks=[]
        for _ in range(2):
            price=np.clip(fare*np.exp(rng.uniform(np.log(.4),np.log(4),len(x))),2.5,70)
            blocks.append((design(z,price).astype('float32'),(price<=wtp).astype(float)))
        theta,info=fit_logit(blocks)
        np.save(DATA/f'{name}.npy',theta)
        A=theta[0]+z@theta[1:21]; B=z@theta[21:]
        info.update(base=base,seconds=time.perf_counter()-t0,accept_share=float(np.mean([b[1].mean() for b in blocks])),
                    nonnegative_price_slope_share=float(np.mean(B>=0)),theta_norm=float(np.linalg.norm(theta)))
        meta['scenarios'][name]=info;print(name,info,flush=True)
    (DATA/'simulator.json').write_text(json.dumps(meta,indent=2)+'\n')


def load_stream(split, stack, T):
    """Native first day, or first seven days overlaid by local time of day. Truncate after ordering."""
    src=SOURCE/'data/processed/cache'
    t=np.load(src/f'{split}_t_ns.npy',mmap_mode='r')
    start=(int(t[0])//DAY)*DAY
    end=np.searchsorted(t,start+stack*DAY)
    pos=np.arange(end)
    if stack>1:
        pos=pos[np.argsort(np.asarray(t[:end])%DAY,kind='stable')]
    pos=pos[:T]
    tt=np.asarray(t[pos]);clock=(tt%DAY if stack>1 else tt-start)/1e9
    clock-=clock[0]
    x=np.array(np.load(src/f'{split}_X.npy',mmap_mode='r')[pos],dtype=float)
    x[:,16]=daylight(tt)
    fare=np.array(np.load(src/f'{split}_fare.npy',mmap_mode='r')[pos],dtype=float)
    return x/np.load(DATA/'scale.npy'),clock,fare,dict(split=split,stack_days=stack,T=len(pos),
        source_first_day=str(np.datetime64(start,'ns')),span_seconds=float(clock[-1]),
        arrival_rate=float((len(pos)-1)/clock[-1]),source_days=int(len(np.unique(tt//DAY))))


def exploration(z,theta,seed):
    from replay import schedule
    idx,trig=schedule(len(z))
    rng=np.random.default_rng(seed)
    price=rng.choice([10.,15.,20.,25.,30.],len(idx))
    phi=design(z[idx],price);y=(rng.random(len(idx))<expit(phi@theta)).astype(float)
    return idx,trig,price,phi,y,np.searchsorted(idx,trig-1,side='right')


def lasso_gap(phi,y,theta,lam):
    """Primal minus a feasible logistic-L1 dual value (a global objective certificate)."""
    u=phi@theta;v=expit(u)-y
    gmax=np.max(np.abs(phi.T@v))
    alpha=np.clip(y+min(1.,lam/max(gmax,1e-300))*v,0.,1.)
    primal=np.sum(np.logaddexp(0,u)-y*u)+lam*np.sum(np.abs(theta))
    dual=-np.sum(xlogy(alpha,alpha)+xlogy(1-alpha,1-alpha))
    return max(0.,float(primal-dual)),float(primal)


def polish_lasso(phi,y,theta,lam):
    """Refine a numerically imprecise conic solution in nonnegative split variables."""
    k=len(theta)
    def objective(v):
        u=phi@(v[:k]-v[k:]);g=phi.T@(expit(u)-y)
        return np.sum(np.logaddexp(0,u)-y*u)+lam*v.sum(),np.r_[g+lam,-g+lam]
    result=minimize(objective,np.r_[np.maximum(theta,0),np.maximum(-theta,0)],
                    jac=True,bounds=[(0,None)]*(2*k),method='L-BFGS-B',
                    options=dict(ftol=1e-14,gtol=1e-8,maxiter=10000,maxls=40))
    return result.x[:k]-result.x[k:]


def learn(method,hp,phi,y,n_at,trig):
    th=np.zeros(41);theta=[];secs=[];internal=[];iters=[];residual=[]
    L=None;last=0;sq=0
    solvers.prox_half_sq_pnorm(np.ones(41),1.,4.) # compile outside measurements
    for j,n in enumerate(n_at):
        t0=time.perf_counter();p=phi[:n];yy=y[:n]
        if method=='pgd':
            sq+=float(np.sum(phi[last:n]**2));Lmax=sq/4
            eta=hp['eta0']*41**(-.25)*np.sqrt(n);eps=hp['eps0']*np.sqrt(41*n)
            th,info=solvers.amle_pgd(p,yy,eta,eps,th,Lmax,L or Lmax,4.)
            L=info['L'];assert info['G']<=eps*(1+1e-8)
        else:
            lam=hp['lam0']*float(trig[j])**.25*np.sqrt(np.log(41)+np.log(trig[j]))
            th,info=solvers.lasso_mosek(p,yy,lam)
            if th is None:
                raise RuntimeError(f'MOSEK returned no finite solution at update {j}')
            gap,obj=lasso_gap(p,yy,th,lam)
            if gap>1e-4*max(1.,obj):
                th=polish_lasso(p,yy,th,lam)
                gap,obj=lasso_gap(p,yy,th,lam)
            if gap>1e-4*max(1.,obj):
                raise RuntimeError(f'MOSEK objective gap {gap/obj:g} at update {j}')
            info['G']=gap/max(1.,obj)
        secs.append(time.perf_counter()-t0);theta.append(th.copy());iters.append(info['iters'])
        internal.append(info['time']);residual.append(info.get('G',np.nan));last=n
        if j%100==0:
            print(method,'update',j,'/',len(trig),'n',n,'sec',round(secs[-1],4),flush=True)
    return dict(thetas=np.array(theta),secs=np.array(secs),internal=np.array(internal),iters=np.array(iters),residual=np.array(residual))


def tag(method,hp):
    return method+''.join(f'_{k}{v:g}' for k,v in sorted(hp.items()))


def fingerprint():
    paths=[ROOT/'run.py',ROOT/'replay.py',ROOT/'solvers.py',DATA/'scale.npy']
    paths+=sorted(DATA.glob('wtp*.npy'))
    inputs=sorted((SOURCE/'data/processed/cache').glob('*_*.npy'))
    metadata=''.join(f'{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}' for p in inputs)
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)+metadata.encode()).hexdigest()


def run(split,stack,T,scenario,seed,method,hp):
    from replay import evaluate
    folder=RESULTS/f'{split}_stack{stack}_T{T}'/scenario/f'seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/(tag(method,hp)+'.npz')
    z,clock,fare,meta=load_stream(split,stack,T)
    true=np.load(DATA/f'{scenario}.npy')
    idx,trig,price,phi,y,n_at=exploration(z,true,seed)
    if out.exists():
        with np.load(out) as saved:
            if 'fingerprint' not in saved or str(saved['fingerprint']) != fingerprint():
                raise RuntimeError(f'Stale cache; archive and rerun: {out}')
            learned={k:saved[k] for k in ['thetas','secs','internal','iters','residual']}
    else:
        learned=learn(method,hp,phi,y,n_at,trig)
        np.savez(out,**learned,n_at=n_at,trig=trig,hp=json.dumps(hp),method=method,
                 meta=json.dumps(meta),load=os.getloadavg()[0],platform=platform.platform(),fingerprint=fingerprint())
    reports=[]
    # One measured timing condition, and preregistered fixed service-time sensitivity.
    for delay in ['measured','zero','1s','5s','30s']:
        seconds=learned['secs'] if delay=='measured' else np.full(len(trig),0 if delay=='zero' else float(delay[:-1]))
        result=evaluate(z,clock,fare,true,idx,price,trig,learned['thetas'],seconds)
        for row in result['summary']:
            reports.append(dict(meta,scenario=scenario,seed=seed,method=method,hp=json.dumps(hp),delay=delay,
                                **row,timely_rate=result['updates']['timely_rate'],
                                solve_total=float(seconds.sum())))
        if delay=='measured':
            np.savez(out.with_name(out.stem+'_curves.npz'),**result['curve'])
            pd.DataFrame({k:result['updates'][k] for k in ['finish','deadline','eligible','timely']}).to_csv(out.with_suffix('.updates.csv'),index=False)
    pd.DataFrame(reports).to_csv(out.with_suffix('.csv'),index=False)
    print('DONE',out,flush=True)
    return reports


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run'])
    ap.add_argument('--split',default='test',choices=['test','val']);ap.add_argument('--stack',type=int,default=1)
    ap.add_argument('--T',type=int,default=300000);ap.add_argument('--scenario',default='wtp16')
    ap.add_argument('--seed',type=int,default=1);ap.add_argument('--method',choices=['pgd','ilqx'],default='pgd')
    ap.add_argument('--eta',type=float,default=.03);ap.add_argument('--eps',type=float,default=.004)
    ap.add_argument('--lam',type=float,default=.03)
    a=ap.parse_args()
    if a.action=='prepare':prepare()
    else:
        hp=dict(eta0=a.eta,eps0=a.eps) if a.method=='pgd' else dict(lam0=a.lam)
        run(a.split,a.stack,a.T,a.scenario,a.seed,a.method,hp)

if __name__=='__main__':main()
