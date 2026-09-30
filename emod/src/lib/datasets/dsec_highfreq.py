from pathlib import Path
import numpy as np
from .dsec_original import OriginalStereoReader,load_calibration,lidar_boxes_to_camera

class DSECHighFrequencyDataset:
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
        if int(a['time_stamp'])!=row['anchor_timestamp_us']:raise ValueError('Anchor timestamp mismatch')
        phase=row['phase'];expected=row['anchor_timestamp_us'] if phase==0 else row['anchor_timestamp_us']+(row['next_keyframe_timestamp_us']-row['anchor_timestamp_us'])*phase//10
        if timestamp!=expected:raise ValueError('Phase query mismatch')
        ann=a['annos'][phase]
        if self.cache_root is None:
            if seq not in self.readers:self.readers[seq]=OriginalStereoReader(self.original_root/seq,self.num_events,self.stack_size)
            events,window=self.readers[seq].read(timestamp)
        else:
            cache=self.cache_root/seq/f'{timestamp}.npz'
            # Cache absence is an error; preprocessing is explicit and separate.
            with np.load(cache,allow_pickle=False) as f:
                if int(f['timestamp_us'])!=timestamp or int(f['num_events'])!=self.num_events or int(f['stack_size'])!=self.stack_size:
                    raise ValueError('Cache timestamp/config mismatch')
                if not int(f['common_start_us'])<=int(f['common_last_us'])<timestamp:raise ValueError('Noncausal cache')
                events={side:f[side] for side in ('left','right')}
                for event_stack in events.values():
                    if event_stack.shape!=(self.stack_size,1,480,640,1) or event_stack.dtype!=np.int8 or not np.isin(event_stack,[-1,0,1]).all():
                        raise ValueError('Invalid DSEC event cache')
            window=None
        events={side:torch.from_numpy(np.pad(e,((0,0),(0,0),(0,0),(0,8),(0,0)))).float() for side,e in events.items()}
        gt=cv2.imread(str(self.original_root/row['disparity_path']),cv2.IMREAD_UNCHANGED) if phase==0 else np.zeros((480,640),dtype=np.uint16)
        if gt is None or gt.dtype!=np.uint16 or gt.shape!=(480,640):raise ValueError('Invalid original event disparity')
        disparity=torch.from_numpy(np.pad(gt.astype(np.float32)/256,((0,0),(0,8))))
        box3d,exact=lidar_boxes_to_camera(ann['gt_boxes_lidar'],cal)
        keep=(exact[:,:,2]>0).all(axis=1)
        box3d,exact=box3d[keep],exact[keep]
        classes=np.array([self.class_mapping[str(n)] for n in ann['name']],dtype=np.int64)[keep]
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
