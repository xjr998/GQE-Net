import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
from plyfile import PlyData, PlyElement

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):

        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'

        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('move %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('ram %s' % (zipfile))


def load_data(partition):

    #download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]



def load_h5(h5txt3D_path):         ###it's incomplete
    with open(h5txt3D_path, "r") as f:
        txt = f.read()
    data1 = []
    label1 = []
    for filename in txt.split():
        h5 = h5py.File(os.path.join(os.path.dirname(h5txt3D_path), filename), 'r')
        data_ = np.array(h5['data'][:, :, :].transpose(2, 1, 0))   # 6,2048,num_sample
        label_ = np.array(h5['label'][:, :, :].transpose(2, 1, 0))
        # qp_ = h5['qp'][:, :]       # num_sample, 1
        # qp_ = np.squeeze(qp_)
        # qp_ = qp_[:, np.newaxis, np.newaxis]
        # qp_ = qp_.repeat(2048, axis=-2)
        # # print(np.size(qp_, 0))
        # # print(np.size(data_, 0))
        # data_ = np.concatenate((data_, qp_), axis=-1)
        print(filename)
        data1.append(data_)
        label1.append(label_)
    data1 = torch.from_numpy((np.concatenate(data1)).astype(np.float32))
    label1 = torch.from_numpy((np.concatenate(label1)).astype(np.float32))
    return data1, label1


class h5Dataset(Dataset):
    def __init__(self, txtPth):
        data, label = load_h5(txtPth)
       # k = random.randrange(1,4)
       # data_aug = np.rot90(data, k, axes=[2, 3])
       # label_aug = np.rot90(label, k, axes=[2, 3])
        self.data = data
      #  print(self.data.shape)
        self.label = label
        self.length = data.shape[0]

    def __getitem__(self, index):
        data = self.data[index, :, :, :]
        label = self.label[index, :, :, :]
       # k = random.randrange(1,5)
       # data = np.rot90(data, k, axes=[1, 2])
       # label = np.rot90(label, k ,axes=[1, 2])
        return data, label

    def __len__(self):
        return self.length


def read_ply(filename):
    """ read XYZ (RGB)point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, red, green, blue] for x, y, z, red, green, blue in pc[['x', 'y', 'z', 'red', 'green', 'blue']]])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3(6), write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5]) for i in range(points.shape[0])]
   # vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
