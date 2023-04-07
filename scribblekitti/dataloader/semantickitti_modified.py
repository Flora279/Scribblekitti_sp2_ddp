import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

city_info = {
    'label2semkitti':
    [[0, 9],      # road
     [1, 11],      # sidewalk
     [2, 13],      # building
     [3, 0],    # wall
     [4, 14],      # fence
     [5, 18],     # pole
     [6, 0],    # traffic light
     [7, 19],     # traffic sign
     [8, 15],      # vegetation
     [9, 17],     # terrain,
     [10, 0],   # sky
     [11, 6],     # person
     [12, 0],   # rider -> bicyclist, weak lbl
     [13, 1],     # car
     [14, 4],     # truck
     [15, 5],   # bus    -> other-vehicle
     [16, 5],   # train  -> other-vehicle
     [17, 3],     # motorcycle
     [18, 2],     # bicycle
    ]
}
city_mapping = np.array(city_info['label2semkitti'], dtype=np.int64)
city_map_vector = np.zeros((city_mapping.shape[0],), dtype=np.int64)
for original_label, train_label in city_mapping:
    city_map_vector[original_label] = train_label

class SemanticKITTI(Dataset):
    _registry = {}
    def __init_subclass__(cls, prefix, **kwargs):   # initialize all subclasses of SemanticKITTI

        super().__init_subclass__(**kwargs)
        cls._registry[prefix] = cls

    def __new__(cls, split:str, config:dict):
        subclass = cls._registry[config['prefix']]
        obj = object.__new__(subclass)
        return obj   # if __new__ return 实例对象, automatically call subclass's __init__()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Baseline(SemanticKITTI, prefix='baseline'):
    def __init__(self, split, config):
        self.split, self.config = split, config
        self.root_dir = self.config['root_dir']
        assert(os.path.isdir(self.root_dir))
        label_directory = 'InternImage' if 'label_directory' not in config.keys() \
                                      else config['label_directory']
        self.label_directory = label_directory if split == 'train' else 'labels'
        self.load_file_paths(split, self.label_directory)

    def load_file_paths(self, split='train', label_directory='labels'):
        self.lidar_paths = []
        self.label_paths = []
        for seq in self.config['split'][split]:
            seq = '{0:02d}'.format(int(seq))
            
            lidar_dir = os.path.join(self.root_dir, seq, 'velodyne')
            lidar_paths = [os.path.join(dp, f) for dp, dn, fn in
                           os.walk(os.path.expanduser(lidar_dir))
                           for f in fn if f.endswith('.bin')]
            self.lidar_paths.extend(lidar_paths)

            if self.label_directory == 'InternImage':
                # get InternImage prediction of common (2d-3d) img pixels
                label_dir = '/home/jzhang2297/data/KITTI_Odometry/dataset/sequences/{0}/image_2_pred/'.format(
                    seq)
                label_paths = [os.path.join(dp, f) for dp, dn, fn in
                               os.walk(os.path.expanduser(label_dir))
                               for f in fn if f.endswith('.png')]


            else:      # else self.label_directory == 'labels'
                label_dir = os.path.join(self.root_dir, seq, label_directory)
                label_paths = [os.path.join(dp, f) for dp, dn, fn in
                           os.walk(os.path.expanduser(label_dir))
                           for f in fn if f.endswith('.label')]
            assert(len(lidar_paths) == len(label_paths))
            self.label_paths.extend(label_paths)
        self.lidar_paths.sort()
        #self.lidar_paths = self.lidar_paths[:48]   # *
        self.label_paths.sort()       # todo: change label path to 90 fov plabel
        #self.label_paths = self.label_paths[:48]     # *

    def __getitem__(self, idx):
        xyzr = self.get_lidar(idx)
        label = self.get_label(idx)
        if self.split == 'train':
            xyzr[:,:3] = self.augment(xyzr[:,:3], self.config['aug'])
        return torch.from_numpy(xyzr), \
               torch.from_numpy(label).squeeze().long()

    def __len__(self):
        return len(self.lidar_paths)

    def get_lidar(self, idx):
        lidar_path = self.lidar_paths[idx]
        lidar = np.fromfile(lidar_path, dtype=np.float32)
        return lidar.reshape((-1, 4))

    def get_label(self, idx):
        label_path = self.label_paths[idx]     # get scribble labels
        seq = label_path.split('/')[-3]
        scan = label_path.split('/')[-1].split('.')[0]
          # set train & val here

        if self.label_directory == 'InternImage':
            path = '/home/jzhang2297/weaksup/mapped_points/{0}_{1}.bin_mapped_pts.npy'.format(seq, scan)
            mask_path = '/home/jzhang2297/weaksup/mapped_points/{0}_{1}.bin_mask.npy'.format(seq, scan)
            map_pts = np.load(path)  # 90 fov 3d points in 2d coordinates, around 2w num per scan
            mask = np.load(mask_path)  # indicate which points are projected to image 90 fov
            # print('mask shape', mask.shape[-1], np.sum(mask))
            labels = np.zeros(mask.shape[-1])
            # use floor operation to get projected 2d coordinates
            h = np.floor(map_pts[:, 0]).astype(int)
            w = np.floor(map_pts[:, 1]).astype(int)
            intern_pred_path = label_path
            intern_pred = cv2.imread(intern_pred_path,
                                     cv2.IMREAD_GRAYSCALE)  # get prediction in cityscapes label id: 0~18
            # print('intern_pred', np.unique(intern_pred))
            common_pixels_pred = city_map_vector[
                intern_pred[h, w].astype(np.int64, copy=False)]  # map city_pred to semkitti id for training.
            # print('common_pixels_pred ', np.unique(common_pixels_pred))
            labels[mask] = common_pixels_pred     # 0~19
            mapped_labels = labels
            # print('get internimage pred plabels', mapped_labels.shape, np.unique(mapped_labels))
        else:            # get full gt labels
            label = np.fromfile(label_path, dtype=np.int32)
            # print('full gt label validation label_path', label_path)
            label = label.reshape((-1)) & 0xFFFF
            mapped_labels = self.map_label(label, self.config['learning_map'])
        return mapped_labels

    @staticmethod
    def map_label(label, map_dict):
        maxkey = 0
        for key, data in map_dict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in map_dict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        return lut[label]

    @staticmethod
    def augment(xyz, methods):
        if 'rotate' in methods:
            angle = np.deg2rad(np.random.random()*90) - np.pi/4
            c, s = np.cos(angle), np.sin(angle)
            R = np.matrix([[c, s], [-s, c]])
            xyz[:,:2] = np.dot(xyz[:,:2], R)

        if 'flip' in methods:
            direction = np.random.choice(4,1)
            if direction == 1:
                xyz[:,0] = -xyz[:,0]
            elif direction == 2:
                xyz[:,1] = -xyz[:,1]
            elif direction == 3:
                xyz[:,:2] = -xyz[:,:2]

        if 'scale' in methods:
            s = np.random.uniform(0.95, 1.05)
            xyz[:,:2] = s * xyz[:,:2]

        if 'noise' in methods:
            noise = np.array([np.random.normal(0, 0.1, 1),
                              np.random.normal(0, 0.1, 1),
                              np.random.normal(0, 0.1, 1)]).T
            xyz[:,:3] += noise
        return xyz


class Cylindrical(Baseline, prefix='cylindrical'):
    def __init__(self, split, config):
        super().__init__(split, config)
        self.spatial_shape = np.array(self.config['spatial_shape'])
        self.max_bound = np.asarray(self.config['max_bound'])
        self.min_bound = np.asarray(self.config['min_bound'])
        self.drpz = (self.max_bound - self.min_bound)/(self.spatial_shape - 1)
        self.label_voxel_zeros = np.zeros(self.spatial_shape, dtype=np.uint8)

    def __getitem__(self, idx):
        xyzr = self.get_lidar(idx)
        label = self.get_label(idx)
        return self.get_cylindrical_scene(xyzr, label, self.config['aug'])

    @staticmethod
    def cart2cyl(xyz):
        rho = np.sqrt(xyz[:,0] ** 2 + xyz[:,1] ** 2)
        phi = np.arctan2(xyz[:,1], xyz[:,0])
        return np.stack((rho, phi, xyz[:,2]), axis=1)

    def get_cylindrical_scene(self, xyzr, label, aug_methods):
        xyz, intensity = xyzr[:,:3], xyzr[:,3]
        if self.split == 'train':
            xyz = self.augment(xyz, aug_methods)

        rpz = self.cart2cyl(xyz)
        clipped_rpz = np.clip(rpz, self.min_bound, self.max_bound)
        rpz_discrete = (np.floor((clipped_rpz - self.min_bound)/self.drpz)).astype(np.int)

        center = (rpz_discrete.astype(np.float32) + 0.5) * self.drpz + self.min_bound
        centered_rpz = rpz - center
        fea = np.concatenate((centered_rpz, rpz, xyz[:,:2], intensity.reshape(-1,1)), axis=1)
        return torch.from_numpy(rpz_discrete), \
               torch.from_numpy(fea).float(), \
               torch.from_numpy(label).squeeze().long()


class CylindricalMT(Cylindrical, prefix='cylindrical_mt'):
    def __getitem__(self, idx):
        xyzr = self.get_lidar(idx)
        label = self.get_label(idx)
        return {
            'student': self.get_cylindrical_scene(xyzr, label, self.config['aug']['student']),
            'teacher': self.get_cylindrical_scene(xyzr, label, self.config['aug']['teacher'])
        }


class PLSCylindricalMT(CylindricalMT, prefix='pls_cylindrical_mt'):
    def __init__(self, split, config, nclasses=20):
        super().__init__(split, config)
        self.load_file_paths('train', self.label_directory)
        self.nclasses = nclasses
        self.bin_sizes = self.config['bin_size']


    def get_cylindrical_scene(self, xyzr, label, aug_methods):
        xyz, intensity = xyzr[:,:3], xyzr[:,3]
        if self.split == 'train':
            xyz = self.augment(xyz, aug_methods)

        rpz = self.cart2cyl(xyz)
        clipped_rpz = np.clip(rpz, self.min_bound, self.max_bound)
        rpz_discrete = (np.floor((clipped_rpz - self.min_bound)/self.drpz)).astype(np.int)

        center = (rpz_discrete.astype(np.float32) + 0.5) * self.drpz + self.min_bound
        centered_rpz = rpz - center
        
        fea = np.concatenate((centered_rpz, rpz, xyz[:,:2], intensity.reshape(-1,1)), axis=1)
        fea = np.concatenate((fea, self.pls(rpz_discrete[:,:2], label)), axis=1)
        # torch.from_numpy(rpz_discrete).shape) # [122765,3]
        # torch.from_numpy(fea).float().shape)   # [122765,66]
        # torch.from_numpy(label).squeeze().long().shape) [122765]
        return torch.from_numpy(rpz_discrete), \
               torch.from_numpy(fea).float(), \
               torch.from_numpy(label).squeeze().long()

    def pls(self, rp_discrete, label):
        N = rp_discrete.shape[0]
        pyramid_semantic_context = np.zeros((N,len(self.bin_sizes),self.nclasses-1))
        for i, bin_size in enumerate(self.bin_sizes):
            rp_coarse = rp_discrete // bin_size
            rp_unique = np.vstack(list({tuple(e) for e in rp_coarse}))

            local_semantic_context = np.zeros((N,self.nclasses-1))
            for key in rp_unique:
                mask = (rp_coarse == key).all(1)
                local_label = label[mask]
                hist = np.histogram(local_label, bins=self.nclasses, range=[0,self.nclasses])[0][1:]
                hist = hist / hist.sum() if hist.sum() > 0 else hist
                local_semantic_context[mask] = np.repeat(hist[None], mask.sum(), 0)
            pyramid_semantic_context[:,i] = local_semantic_context
        return pyramid_semantic_context.reshape(N,-1)