"""Original DSEC stereo reader in microseconds; never applies SE3D time scaling.

Fixed count, causal raw events; rectify then intersect the two temporal windows,
matching the existing mixed-density stacking convention. Files stay read-only.
"""
from pathlib import Path
import h5py
import hdf5plugin
import numpy as np
import yaml
from .dsec.event.sbn.stack import MixedDensityEventStacking

class OriginalEventReader:
    def __init__(self, path, num_events=5_000_000):
        self.path=Path(path)
        self.num_events=int(num_events)
        with h5py.File(self.path/'events.h5','r') as f:
            self.offset=int(f['t_offset'][()])
            self.index=f['ms_to_idx'][:].astype(np.int64)
            self.count=len(f['events/t'])
            self.first=int(f['events/t'][0])+self.offset
            self.last=int(f['events/t'][-1])+self.offset
        with h5py.File(self.path/'rectify_map.h5','r') as f:
            self.rectify_map=f['rectify_map'][:]
        if self.rectify_map.shape!=(480,640,2):raise ValueError('Unexpected rectification map shape')

    def read(self, timestamp):
        timestamp=int(timestamp)
        if not self.first<timestamp<=self.last:raise ValueError(f'Timestamp outside event stream: {timestamp}')
        relative=timestamp-self.offset
        ms=relative//1000
        lo=int(self.index[ms]) if ms<len(self.index) else self.count
        hi=int(self.index[ms+1]) if ms+1<len(self.index) else self.count
        with h5py.File(self.path/'events.h5','r') as f:
            end=lo+int(np.searchsorted(f['events/t'][lo:hi],relative,side='left'))
            if end<self.num_events:raise ValueError(f'Insufficient causal history: {end} < {self.num_events}')
            start=end-self.num_events
            a={k:f['events/'+k][start:end] for k in ('x','y','p','t')}
        a['t']=a['t'].astype(np.int64)+self.offset
        if len(a['t'])!=self.num_events or a['t'][-1]>=timestamp:raise ValueError('Causal slice contract violated')
        xy=self.rectify_map[a['y'],a['x']]
        valid=(xy[:,0]>=0)&(xy[:,0]<640)&(xy[:,1]>=0)&(xy[:,1]<480)
        # Existing stacker truncates positive float coordinates to integer pixels.
        return dict(x=xy[valid,0],y=xy[valid,1],p=a['p'][valid],t=a['t'][valid])

class OriginalStereoReader:
    def __init__(self, sequence_root, num_events=5_000_000, stack_size=10):
        self.root=Path(sequence_root)
        self.readers={side:OriginalEventReader(self.root/'events'/side,num_events) for side in ('left','right')}
        self.stack=MixedDensityEventStacking(stack_size,num_events,480,640)
        self.stack_size=stack_size

    def read(self,timestamp):
        events={side:r.read(timestamp) for side,r in self.readers.items()}
        if any(not len(a['t']) for a in events.values()):raise ValueError('No rectified events')
        start=max(a['t'][0] for a in events.values());stop=min(a['t'][-1] for a in events.values())
        if stop<start:raise ValueError('No common stereo time window')
        output={}
        for side,a in events.items():
            mask=(a['t']>=start)&(a['t']<=stop)
            a={k:v[mask] for k,v in a.items()}
            if not len(a['t']):raise ValueError('Empty stereo slice')
            sparse=self.stack.pre_stack(a,int(timestamp))
            output[side]=self.stack.post_stack(sparse).transpose(4,0,1,2,3)
        return output,dict(timestamp_us=int(timestamp),common_start_us=int(start),common_last_us=int(stop))


def load_calibration(sequence_root, annotation_event_to_lidar=None):
    p=Path(sequence_root)/'calibration'
    c=yaml.safe_load((p/'cam_to_cam.yaml').read_text())
    lidar=yaml.safe_load((p/'cam_to_lidar.yaml').read_text())
    e=c['extrinsics'];r0=np.eye(4);r1=np.eye(4)
    r0[:3,:3]=e['R_rect0'];r1[:3,:3]=e['R_rect1']
    event_to_lidar=np.array(lidar['T_lidar_camRect1'])@r1@np.array(e['T_10'])@np.linalg.inv(r0)
    fx,fy,cx,cy=c['intrinsics']['camRect0']['camera_matrix']
    left=np.array([[fx,0,cx,0],[0,fy,cy,0],[0,0,1,0.]])
    baseline=1/np.array(c['disparity_to_depth']['cams_03'])[3,2]
    right=left.copy();right[0,3]=-fx*baseline
    physical_event_to_lidar=event_to_lidar.copy()
    if annotation_event_to_lidar is not None:
        event_to_lidar=np.asarray(annotation_event_to_lidar,dtype=np.float64)
        if event_to_lidar.shape!=(4,4) or not np.isfinite(event_to_lidar).all():raise ValueError('Invalid annotation extrinsic')
        if not np.allclose(event_to_lidar[3],[0,0,0,1]) or not np.allclose(event_to_lidar[:3,:3]@event_to_lidar[:3,:3].T,np.eye(3),atol=1e-6):raise ValueError('Annotation extrinsic is not rigid')
    return dict(left=left,right=right,event_to_lidar=event_to_lidar,lidar_to_event=np.linalg.inv(event_to_lidar),physical_event_to_lidar=physical_event_to_lidar,coordinate_contract='provided_annotation_frame' if annotation_event_to_lidar is not None else 'original_yaml_physical_frame',baseline=float(baseline),focal=float(fx),Q=np.array(c['disparity_to_depth']['cams_03']))


def lidar_boxes_to_camera(boxes, calibration):
    """Return KITTI h,w,l,x,y_bottom,z,ry and exact transformed eight corners.

Loss uses yaw-only boxes; exact corners are retained to quantify pitch/roll
approximation and for projection checks. Source LiDAR labels remain unchanged.
"""
    boxes=np.asarray(boxes)
    if not len(boxes):return np.empty((0,7)),np.empty((0,8,3))
    t=calibration['lidar_to_event']
    centers=boxes[:,:3]@t[:3,:3].T+t[:3,3]
    c,s=np.cos(boxes[:,6]),np.sin(boxes[:,6])
    direction=np.stack((c,s,np.zeros(len(boxes))),axis=-1)@t[:3,:3].T
    yaw=np.arctan2(-direction[:,2],direction[:,0])
    size=boxes[:,[5,4,3]]
    bottom=centers.copy();bottom[:,1]+=size[:,0]/2
    result=np.concatenate((size,bottom,yaw[:,None]),axis=1)
    signs=np.array([[-1,-1,-1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,1],[-1,1,1],[1,1,1],[1,-1,1]])
    local=signs[None]*boxes[:,None,3:6]/2
    x=local[:,:,0]*c[:,None]-local[:,:,1]*s[:,None]
    y=local[:,:,0]*s[:,None]+local[:,:,1]*c[:,None]
    corners=np.stack((x,y,local[:,:,2]),axis=-1)+boxes[:,None,:3]
    exact=corners@t[:3,:3].T+t[:3,3]
    return result,exact

class DSECKeyframeDataset:
    """Frozen paired keyframes; no data-dependent skipping or implicit caching."""
    HEIGHT=480
    WIDTH=640
    def __init__(self, manifest_path, split, generate_target=True, cache_root=None):
        import json
        from configs.od_cfg import cfg
        manifest=__import__('json').loads(Path(manifest_path).read_text())
        if split not in manifest['splits']:raise ValueError(split)
        self.rows=manifest['splits'][split]
        self.original_root=Path(manifest['original_root']);self.labels_root=Path(manifest['labels_root'])
        self.num_events=manifest['num_past_raw_events'];self.stack_size=manifest['stack_size']
        self.generate_target=generate_target
        self.cache_root=Path(cache_root) if cache_root else None
        self.cfg=cfg
        self.readers={};self.calibrations={};self.annotations={};self.builders={}
        expected=['Vehicle','Pedestrian','Cyclist']
        if list(getattr(cfg,'class_names',[]))!=expected:raise ValueError('Configure DSEC Vehicle/Pedestrian/Cyclist before dataset creation')
        self.class_mapping={n:i+1 for i,n in enumerate(expected)}

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        import pickle
        import cv2
        import torch
        from types import SimpleNamespace
        from .dsec.labels.base.dataset import LabelsDataset
        from lib.dsgn.utils.bounding_box import Box3DList
        row=self.rows[index];seq=row['sequence'];chunk=row['chunk'];timestamp=row['timestamp_us']
        if chunk not in self.annotations:self.annotations[chunk]=pickle.loads((self.labels_root/row['annotation_path']).read_bytes())
        a=self.annotations[chunk][row['frame']]
        if seq not in self.calibrations:self.calibrations[seq]=load_calibration(self.original_root/seq,a['image']['image_0_extrinsic'])
        cal=self.calibrations[seq]
        if not np.allclose(cal['event_to_lidar'],a['image']['image_0_extrinsic'],atol=1e-10,rtol=0):raise ValueError('Annotation calibration varies within sequence')
        if int(a['time_stamp'])!=timestamp:raise ValueError('Annotation no longer matches frozen timestamp')
        if self.cache_root is None:
            if seq not in self.readers:self.readers[seq]=OriginalStereoReader(self.original_root/seq,self.num_events,self.stack_size)
            events,window=self.readers[seq].read(timestamp)
        else:
            cache=self.cache_root/seq/f'{timestamp}.npz'
            # Cache absence is an error; preprocessing is explicit and separate.
            with np.load(cache,allow_pickle=False) as f:
                if int(f['timestamp_us'])!=timestamp or int(f['num_events'])!=self.num_events or int(f['stack_size'])!=self.stack_size:
                    raise ValueError('Cache timestamp/config mismatch')
                events={side:f[side] for side in ('left','right')}
                for event_stack in events.values():
                    if event_stack.shape!=(self.stack_size,1,480,640,1) or event_stack.dtype!=np.int8 or not np.isin(event_stack,[-1,0,1]).all():
                        raise ValueError('Invalid DSEC event cache')
            window=None
        events={side:torch.from_numpy(np.pad(e,((0,0),(0,0),(0,0),(0,8),(0,0)))).float() for side,e in events.items()}
        gt=cv2.imread(str(self.original_root/row['disparity_path']),cv2.IMREAD_UNCHANGED)
        if gt is None or gt.dtype!=np.uint16 or gt.shape!=(480,640):raise ValueError('Invalid original event disparity')
        disparity=torch.from_numpy(np.pad(gt.astype(np.float32)/256,((0,0),(0,8))))
        box3d,exact=lidar_boxes_to_camera(a['annos']['gt_boxes_lidar'],cal)
        keep=(exact[:,:,2]>0).all(axis=1)
        box3d,exact=box3d[keep],exact[keep]
        classes=np.array([self.class_mapping[str(n)] for n in a['annos']['name']],dtype=np.int64)[keep]
        if len(exact):
            projected=exact@cal['left'][:,:3].T
            pixels=projected[:,:,:2]/projected[:,:,2:3]
            bbox=np.concatenate((pixels.min(axis=1),pixels.max(axis=1)),axis=1)
            bbox[:,[0,2]]=bbox[:,[0,2]].clip(0,639);bbox[:,[1,3]]=bbox[:,[1,3]].clip(0,479)
            keep=(bbox[:,2]>bbox[:,0])&(bbox[:,3]>bbox[:,1])
            bbox,box3d,classes=bbox[keep],box3d[keep],classes[keep]
            order=np.lexsort((-box3d[:,5],classes));bbox,box3d,classes=bbox[order],box3d[order],classes[order]
        else:bbox=np.empty((0,4))
        box3d=torch.as_tensor(box3d,dtype=torch.float32)
        if self.cfg.learn_viewpoint and len(box3d):box3d[:,6]+=torch.atan2(box3d[:,5],box3d[:,3])-np.pi/2
        target=Box3DList(torch.as_tensor(bbox,dtype=torch.float32),(640,480),mode='xyxy',box3d=box3d,Proj=cal['left'],Proj_R=cal['right'])
        target.add_field('labels',torch.as_tensor(classes,dtype=torch.long))
        if seq not in self.builders:
            builder=LabelsDataset.__new__(LabelsDataset)
            builder.cfg=self.cfg;builder.valid_classes=list(self.cfg.valid_classes);builder.class_mapping=self.class_mapping
            builder.generate_target=self.generate_target;builder._anchors={}
            builder.calib=SimpleNamespace(P=cal['left']);builder.calib_R=SimpleNamespace(P=cal['right'])
            self.builders[seq]=builder
        builder=self.builders[seq];iou,label_map=builder.target_maps(target)
        labels=[None,row['frame'],(640,480),iou,label_map,builder.calib,builder.calib_R,target]
        return dict(event=events,disparity=disparity,labels=labels,file_index=index,metadata=row)

    @staticmethod
    def collate_fn(batch):
        import torch
        return dict(event={side:torch.stack([b['event'][side] for b in batch]) for side in ('left','right')},
                    disparity=torch.stack([b['disparity'] for b in batch]),labels=[b['labels'] for b in batch],
                    file_index=torch.tensor([b['file_index'] for b in batch]),metadata=[b['metadata'] for b in batch])
