"""
|---dataset
|  |---dsec
|  |  |---bag_name
|  |  |  |---blurry
|  |  |  |---events
|  |  |  |---sharps
|  |  |  |---disps
"""
import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utilities.event_reader import EventReader


class StDataLoader(Dataset):
    def __init__(self, args, split="train"):
        super(StDataLoader, self).__init__()
        self.args = args
        self.split = split
        self.bag_names, self.num_group = self.readFilePaths()
        self.event_reader_dsec = EventReader(480, 640, num_bins=6, normalize=False)
        self.event_reader_mvsec = EventReader(260, 346, num_bins=6, normalize=False)

    def __len__(self):
        return self.num_group[-1]

    def __getitem__(self, idx):
        bag_iter = 0
        for i in self.num_group:
            if idx < i:
                break
            bag_iter += 1
        bag_iter -= 1

        bag_name = self.bag_names[bag_iter]
        image_iter = (idx - self.num_group[bag_iter]) * 6
        
        """ -------------------- start reading data -------------------- """
        blurry_dir = os.path.join(self.args.dataset_directory, bag_name, 'blurry')
        events_dir = os.path.join(self.args.dataset_directory, bag_name, 'events')
        
        """ -------------------- load blurry -------------------- """
        blurry_path = os.path.join(blurry_dir, str(image_iter).rjust(5, '0') + '.png')
        blurry = np.transpose(cv2.imread(blurry_path), (2, 0, 1))  # [3, h, w]

        """ -------------------- load events -------------------- """
        event_path_01 = os.path.join(events_dir, str(image_iter + 0).rjust(5, '0') + '.h5')
        event_path_12 = os.path.join(events_dir, str(image_iter + 1).rjust(5, '0') + '.h5')
        event_path_23 = os.path.join(events_dir, str(image_iter + 2).rjust(5, '0') + '.h5')
        event_path_34 = os.path.join(events_dir, str(image_iter + 3).rjust(5, '0') + '.h5')
        event_path_45 = os.path.join(events_dir, str(image_iter + 4).rjust(5, '0') + '.h5')
        event_path_56 = os.path.join(events_dir, str(image_iter + 5).rjust(5, '0') + '.h5')

        if 'dsec' in bag_name:
            event_voxelgrid_01 = self.event_reader_dsec.h5Reader_dsec(event_path_01)  # [bins, h, w]
            event_voxelgrid_12 = self.event_reader_dsec.h5Reader_dsec(event_path_12)
            event_voxelgrid_23 = self.event_reader_dsec.h5Reader_dsec(event_path_23)
            event_voxelgrid_34 = self.event_reader_dsec.h5Reader_dsec(event_path_34)
            event_voxelgrid_45 = self.event_reader_dsec.h5Reader_dsec(event_path_45)
            event_voxelgrid_56 = self.event_reader_dsec.h5Reader_dsec(event_path_56)
        elif 'mvsec' in bag_name:
            event_voxelgrid_01 = self.event_reader_mvsec.h5Reader_mvsec(event_path_01)  # [bins, h, w]
            event_voxelgrid_12 = self.event_reader_mvsec.h5Reader_mvsec(event_path_12)
            event_voxelgrid_23 = self.event_reader_mvsec.h5Reader_mvsec(event_path_23)
            event_voxelgrid_34 = self.event_reader_mvsec.h5Reader_mvsec(event_path_34)
            event_voxelgrid_45 = self.event_reader_mvsec.h5Reader_mvsec(event_path_45)
            event_voxelgrid_56 = self.event_reader_mvsec.h5Reader_mvsec(event_path_56)
        elif 'steic' in bag_name:
            event_voxelgrid_01 = self.event_reader_dsec.h5Reader_steic(event_path_01)  # [bins, h, w]
            event_voxelgrid_12 = self.event_reader_dsec.h5Reader_steic(event_path_12)
            event_voxelgrid_23 = self.event_reader_dsec.h5Reader_steic(event_path_23)
            event_voxelgrid_34 = self.event_reader_dsec.h5Reader_steic(event_path_34)
            event_voxelgrid_45 = self.event_reader_dsec.h5Reader_steic(event_path_45)
            event_voxelgrid_56 = self.event_reader_dsec.h5Reader_steic(event_path_56)
        
        """ -------------------- random crop -------------------- """
        if blurry.shape[1] > self.args.crop_height and blurry.shape[2] > self.args.crop_width:
            if self.split == "train":
                y = np.random.randint(low=1, high=(blurry.shape[1] - self.args.crop_height))
                x = np.random.randint(low=1, high=(blurry.shape[2] - self.args.crop_width))
            else:
                y = (blurry.shape[1] - self.args.crop_height) // 2
                x = (blurry.shape[2] - self.args.crop_width) // 2

            blurry = blurry[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            event_voxelgrid_01 = event_voxelgrid_01[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            event_voxelgrid_12 = event_voxelgrid_12[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            event_voxelgrid_23 = event_voxelgrid_23[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            event_voxelgrid_34 = event_voxelgrid_34[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            event_voxelgrid_45 = event_voxelgrid_45[..., y:y + self.args.crop_height, x:x + self.args.crop_width]
            event_voxelgrid_56 = event_voxelgrid_56[..., y:y + self.args.crop_height, x:x + self.args.crop_width]

        """ -------------------- to tensor -------------------- """
        blurry = torch.from_numpy(blurry).float() / 255.
        event_voxelgrid_01 = torch.from_numpy(event_voxelgrid_01).float()
        event_voxelgrid_12 = torch.from_numpy(event_voxelgrid_12).float()
        event_voxelgrid_23 = torch.from_numpy(event_voxelgrid_23).float()
        event_voxelgrid_34 = torch.from_numpy(event_voxelgrid_34).float()
        event_voxelgrid_45 = torch.from_numpy(event_voxelgrid_45).float()
        event_voxelgrid_56 = torch.from_numpy(event_voxelgrid_56).float()

        """ -------------------- mean std normalization -------------------- """
        evgs_06 = torch.cat((event_voxelgrid_01, event_voxelgrid_12, event_voxelgrid_23, event_voxelgrid_34, event_voxelgrid_45, event_voxelgrid_56), 0)

        return blurry, evgs_06, bag_name, image_iter


    def readFilePaths(self):
        bag_names = []
        
        bag_list_file = open(os.path.join(self.args.dataset_directory, self.split + ".txt"))
        lines = bag_list_file.read().splitlines()
        bag_list_file.close()
        
        num_group = [0]
        for line in lines:
            bag_name = line
            bag_names.append(bag_name)
            
            blurry_path = os.path.join(self.args.dataset_directory, bag_name, "blurry")
            blurry_files = os.listdir(blurry_path)
            blurry_num = len(blurry_files)
            
            num_group.append(blurry_num + num_group[-1])
            
        return bag_names, num_group
