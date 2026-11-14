#!/usr/bin/env python3
"""Re-evaluate archived trajectories using raw IMU only; no site/truth files read."""
import argparse, json, hashlib, sqlite3
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "eval/no_truth"
ALGOS = ['GLIM(map)', 'GLIM(odom)', 'FAST-LIO2', 'LIO-SAM', 'Point-LIO']
PROFILES = {'strict': (.04, .025), 'primary': (.06, .04), 'loose': (.08, .05)}
# Route-return labels from acquisition descriptions, without coordinates or lengths.
RETURN_RUNS = {'01_1', '01_2', '02_1', '02_2', '03_01', '04_01'}

def smooth(x, n):
    left = n // 2
    return np.convolve(np.pad(x, (left, n-1-left), mode='edge'), np.ones(n)/n, mode='valid')

def imu_load(bag):
    rows=[]; counts={}
    for p in sorted(bag.glob('*.db3')):
        db=sqlite3.connect(f'file:{p}?mode=ro', uri=True)
        topics=db.execute('SELECT id,name FROM topics').fetchall()
        counts.update({name:db.execute('SELECT count(*) FROM messages WHERE topic_id=?',(i,)).fetchone()[0] for i,name in topics})
        ids=[i for i,name in topics if name=='/livox/imu']
        for i in ids:
            for (blob,) in db.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',(i,)):
                m=deserialize_message(blob, Imu)
                rows.append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
                    m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z,
                    m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z])
        db.close()
    a=np.array(rows); a=a[np.argsort(a[:,0])]
    return a,counts

def windows(a, gthr, athr):
    t=a[:,0]; n=max(1,round(.1/np.median(np.diff(t))))
    gyro=smooth(np.linalg.norm(a[:,1:4],axis=1), n)
    acc=smooth(np.linalg.norm(a[:,4:7],axis=1), n)
    # Livox IMUs in these bags express acceleration in g or SI units.
    unit=1.0 if .8<np.median(acc)<1.2 else 9.80665
    mask=(gyro<gthr)&(np.abs(acc/unit-1)<athr)
    edges=np.diff(np.r_[False,mask,False].astype(int))
    intervals=[]
    for s,e in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
        if t[e-1]-t[s]>=3.0:
            intervals.append([float(t[s]+.5),float(t[e-1]-.5)])
    return intervals, mask, unit

def stats(a, wins):
    vals=[]
    for i,(lo,hi) in enumerate(wins):
        p=a[(a[:,0]>=lo)&(a[:,0]<=hi)]
        if len(p)<5 or p[-1,0]-p[0,0] < .8*(hi-lo):
            vals.append(None);continue
        dev=p[:,1:4]-p[:,1:4].mean(0)
        x,y,z,w=p[:,4:8].T
        yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        vals.append(dict(index=i,n=len(p),xy_mm=float(np.sqrt(np.mean(np.sum(dev[:,:2]**2,axis=1)))*1000),
             xyz_mm=float(np.sqrt(np.mean(np.sum(dev**2,axis=1)))*1000),yaw_deg=float(np.std(yaw)*180/np.pi)))
    return vals

def align(P,Q):
    pc=P-P.mean(0); qc=Q-Q.mean(0)
    u,s,vt=np.linalg.svd(pc.T@qc)
    d=np.eye(3); d[-1,-1]=np.linalg.det(u@vt)
    r=u@d@vt
    return pc@r+Q.mean(0)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag-root', type=Path, default=PROJECT / 'data/rosbag',
                        help='原始 bag 的父目录，默认 data/rosbag/；各子目录名须与清单序列一致')
    args = parser.parse_args()
    bag_root = args.bag_root.expanduser().resolve()
    manifest=json.loads((PROJECT/'config/manifest.json').read_text())['bags']
    missing = [bag for bag in manifest if not list((bag_root/bag).glob('*.db3'))]
    if missing:
        parser.error(f'原始数据目录 {bag_root} 缺少 SQLite3 bag：{", ".join(missing)}。'
                     '请使用 --bag-root 指向包含这些序列的目录。')
    OUT.mkdir(parents=True, exist_ok=True)
    result={'method':{'imu_profiles':PROFILES,'smoothing_s':.1,'min_raw_window_s':3,'trim_each_side_s':.5,
          'grid_s':.1,'return_sequences':sorted(RETURN_RUNS),'truth_files_read':False,
          'divergence_gate':'p99 speed > 5 m/s OR coordinate span > 100 m',
          'execution':'Raw-IMU window extraction and trajectory metrics rerun; algorithm trajectories reused.'},'runs':{},'source_files':[]}
    for bag,info in manifest.items():
        imu,counts=imu_load(bag_root/bag)
        variants={}; masks={}
        for key,(gthr,athr) in PROFILES.items():
            variants[key],masks[key],unit=windows(imu,gthr,athr)
        trajs={}
        for name,rel in info['traj'].items():
            p=PROJECT/rel
            a=np.loadtxt(p); a=a[np.argsort(a[:,0])]
            if not np.isfinite(a).all():raise ValueError(p)
            if np.any(np.diff(a[:,0])<=0):raise ValueError('duplicate timestamp: '+str(p))
            trajs[name]=a
            result['source_files'].append({'path':str(p.relative_to(PROJECT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        lo=max(a[0,0] for a in trajs.values());hi=min(a[-1,0] for a in trajs.values())
        common_wins={k:[[max(x,lo),min(y,hi)] for x,y in ws if min(y,hi)-max(x,lo)>=2] for k,ws in variants.items()}
        # This common set is used by every output, including numerical failures.
        run={'radar':info['radar'],'imu_messages':len(imu),'lidar_messages':counts.get('/livox/lidar'),
             'imu_duration_s':float(imu[-1,0]-imu[0,0]),'imu_start':float(imu[0,0]),
             'acc_unit':unit,'common_span':[float(lo),float(hi)],'windows':common_wins,'algorithms':{},'pairs':[]}
        startmask=(imu[:,0]>=lo)&(imu[:,0]<=lo+1)
        endmask=(imu[:,0]>=hi-1)&(imu[:,0]<=hi)
        endpoint_low=[float(np.mean(masks['primary'][m])) for m in (startmask,endmask)]
        run['endpoint_low_dynamic_fraction']=endpoint_low
        run['closure_defined']=bag in RETURN_RUNS and min(endpoint_low)>=.8
        for name,a in trajs.items():
            dt=np.diff(a[:,0]); steps=np.linalg.norm(np.diff(a[:,1:4],axis=0),axis=1)
            speed=steps/dt; span=np.ptp(a[:,1:4],axis=0)
            failed=np.percentile(speed,99)>5 or max(span)>100
            st={k:stats(a,ws) for k,ws in common_wins.items()}
            r={'poses':len(a),'duration_s':float(a[-1,0]-a[0,0]),'p99_speed_m_s':float(np.percentile(speed,99)),
               'max_step_m':float(max(steps)),'max_step_time_s':float(a[1+np.argmax(steps),0]-imu[0,0]),
               'coordinate_span_m':span.tolist(),'diverged':bool(failed),'window_stats':st}
            first=a[(a[:,0]>=lo)&(a[:,0]<=lo+1),1:4].mean(0)
            last=a[(a[:,0]>=hi-1)&(a[:,0]<=hi),1:4].mean(0)
            r['endpoint_gap_mm']=float(np.linalg.norm(last-first)*1000)
            r['closure_mm']=r['endpoint_gap_mm'] if run['closure_defined'] else None
            r['profile_medians']={k: {metric:float(np.median([v[metric] for v in vs if v is not None])) if any(v is not None for v in vs) else None for metric in ['xy_mm','xyz_mm','yaw_deg']} for k,vs in st.items()}
            run['algorithms'][name]=r
        grid=np.arange(lo,hi,.1)
        interp={name:np.column_stack([np.interp(grid,a[:,0],a[:,i]) for i in [1,2,3]]) for name,a in trajs.items() if not run['algorithms'][name]['diverged']}
        for i,name in enumerate(interp):
            for other in list(interp)[i+1:]:
                err=np.linalg.norm(align(interp[name],interp[other])-interp[other],axis=1)*1000
                run['pairs'].append({'a':name,'b':other,'rms_mm':float(np.sqrt(np.mean(err**2))),'p95_mm':float(np.percentile(err,95)),'max_mm':float(max(err)),'n':len(grid)})
        result['runs'][bag]=run
        print(bag, 'windows', {k:len(v) for k,v in common_wins.items()},'closure',run['closure_defined'],'end',np.round(endpoint_low,2),flush=True)
        for name,r in run['algorithms'].items():
            print(' ',name, 'xy',r['profile_medians']['primary']['xy_mm'],'close',r['closure_mm'],'step',round(r['max_step_m'],3),'FAIL' if r['diverged'] else '',flush=True)
    summary={}
    for name in ALGOS:
        rs=[(bag,run['algorithms'][name]) for bag,run in result['runs'].items() if name in run['algorithms']]
        good=[(bag,r) for bag,r in rs if not r['diverged']]
        summary[name]={'valid_runs':len(good),'diverged':[bag for bag,r in rs if r['diverged']],
            'profiles':{k:{metric:float(np.median([r['profile_medians'][k][metric] for bag,r in good if r['profile_medians'][k][metric] is not None])) for metric in ['xy_mm','xyz_mm','yaw_deg']} for k in PROFILES},
            'closure_n':sum(r['closure_mm'] is not None for bag,r in good),
            'closure_mm':float(np.median([r['closure_mm'] for bag,r in good if r['closure_mm'] is not None]))}
    result['summary']=summary
    (OUT/'metrics_no_truth.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
