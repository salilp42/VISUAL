"""
Dataset implementations for NeuroHoloNet.
"""

import os
import random
import requests
import shutil
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset

def generate_voxel_cube(size=32, margin=4):
    vol = np.zeros((size, size, size), dtype=np.float32)
    s, e = margin, size - margin
    vol[s:e, s:e, s:e] = 1.0
    return vol

def generate_voxel_sphere(size=32, margin=4):
    vol = np.zeros((size, size, size), dtype=np.float32)
    center = (size//2, size//2, size//2)
    radius = (size//2) - margin
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                    vol[x,y,z] = 1.0
    return vol

class CubeSphereDataset(Dataset):
    """Synthetic dataset of half cubes, half spheres."""
    def __init__(self, n_samples=64, size=32, transform=None):
        super().__init__()
        self.n_samples = n_samples
        self.size = size
        self.transform = transform
        self.data = []
        half = n_samples//2
        for _ in range(half):
            c = generate_voxel_cube(size=size)
            s = generate_voxel_sphere(size=size)
            self.data.append((c, 0))
            self.data.append((s, 1))
        random.shuffle(self.data)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        vol_np, label = self.data[idx]
        vol_t = torch.from_numpy(vol_np)
        vol_t = vol_t.unsqueeze(0)
        if self.transform:
            vol_t = self.transform(vol_t)
        return vol_t, torch.tensor(label, dtype=torch.long)

class ModelNet10Dataset(Dataset):
    """Minimal ModelNet10 with .off voxelization."""
    def __init__(self, root="modelnet10", split="train", voxel_size=32,
                 url="http://modelnet.cs.princeton.edu/ModelNet10.zip",
                 transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.url = url
        self.transform = transform
        self._check_and_download()
        self.files = self._collect_files(split)

    def _check_and_download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, "ModelNet10.zip")
        if not os.path.exists(zip_path):
            print("Downloading ModelNet10 ...")
            with requests.get(self.url, stream=True) as r:
                with open(zip_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            print("Download complete.")
        extracted_marker = os.path.join(self.root, "extracted")
        if not os.path.exists(extracted_marker):
            print("Extracting ModelNet10 ...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.root)
            open(extracted_marker,'w').close()
            print("Extraction complete.")

    def _collect_files(self, split):
        base_dir = os.path.join(self.root, "ModelNet10")
        categories = [c for c in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,c))]
        all_files = []
        for cat in categories:
            cat_dir = os.path.join(base_dir, cat)
            sp_dir = os.path.join(cat_dir, split)
            if not os.path.exists(sp_dir):
                continue
            for fname in os.listdir(sp_dir):
                if fname.endswith('.off'):
                    all_files.append((os.path.join(sp_dir, fname), cat))
        return all_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, cat = self.files[idx]
        vox_np = self._voxelize_off(path, self.voxel_size)
        label = self._cat_to_label(cat)
        vox_t = torch.from_numpy(vox_np).unsqueeze(0)
        if self.transform:
            vox_t = self.transform(vox_t)
        return vox_t, torch.tensor(label, dtype=torch.long)

    def _cat_to_label(self, cat):
        return hash(cat)%1000

    def _voxelize_off(self, off_path, size):
        with open(off_path,'r') as f:
            lines = f.read().strip().split()
        idx=1
        nv = int(lines[idx]); nf=int(lines[idx+1])
        idx+=3
        verts=[]
        for _ in range(nv):
            vx=float(lines[idx]); vy=float(lines[idx+1]); vz=float(lines[idx+2])
            idx+=3
            verts.append((vx,vy,vz))
        faces=[]
        for _ in range(nf):
            fsz=int(lines[idx]); idx+=1
            fv=[int(lines[idx+i]) for i in range(fsz)]
            idx+=fsz
            faces.append(fv)

        verts_np = np.array(verts,dtype=np.float32)
        mn=verts_np.min(0); mx=verts_np.max(0)
        rng=mx-mn; rng[rng<1e-8]=1
        verts_np=(verts_np-mn)/rng*(size-1)

        vol=np.zeros((size,size,size),dtype=np.float32)
        for (x,y,z) in verts_np:
            ix,iy,iz=int(x),int(y),int(z)
            ix=np.clip(ix,0,size-1)
            iy=np.clip(iy,0,size-1)
            iz=np.clip(iz,0,size-1)
            vol[ix,iy,iz]=1.0
        return vol
