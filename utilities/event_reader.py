import numpy as np
import h5py
import hdf5plugin
from skimage.morphology import remove_small_objects
from omegaconf import OmegaConf
import cv2

import torch

from utilities.event_zipper import VoxelGrid


class EventReader:
    def __init__(self, height, width, num_bins=1, normalize=True):
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=normalize)
    
    def event2voxelgrid(self, x, y, p, t):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        p = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(p),
                torch.from_numpy(t))

    def rectifyEvents(self, x: np.ndarray, y: np.ndarray, rect_ev_maps):
        # From distorted to undistorted
        rectify_map = rect_ev_maps
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def single_intergral(self, x, y, p, t):
        event_img = np.zeros((self.height, self.width)).ravel()
        np.add.at(event_img, x.astype(np.uint16) + y.astype(np.uint16)*self.width, p)
        event_img = event_img.reshape((1, self.height, self.width))
        return event_img

    def h5Reader_dsec(self, path):
        event_data = h5py.File(path, 'r')
        event_t = np.array(event_data['events/t'])
        event_x = np.array(event_data['events/x'])
        event_y = np.array(event_data['events/y'])
        event_p = np.array(event_data['events/p'])

        confpath = path[:-22]+'/calibration/cam_to_cam.yaml'
        conf = OmegaConf.load(confpath)

        K_3 = np.eye(3)
        K_3[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['cam3']['camera_matrix']
        D_3 = np.array(conf['intrinsics']['cam3']['distortion_coeffs'])

        K_r3 = np.eye(3)
        K_r3[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect3']['camera_matrix']

        T_32 = np.asarray(conf['extrinsics']['T_32'])
        T_r2 = np.eye(4)
        T_r2[:3,:3] = np.asarray(conf['extrinsics']['R_rect2'])
        T_r32 = np.matmul(np.linalg.inv(T_32), T_r2)
        T_r32 = T_r32[:3, :3]

        event_xy = np.stack((event_x, event_y),axis=1).astype(np.float32)
        rect_xy = cv2.undistortPointsIter(event_xy, K_3, D_3, T_r32, K_r3, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.01))
        rect_xy = np.reshape(rect_xy, (-1, 2))
        rect_x = rect_xy[:, 0]
        rect_y = rect_xy[:, 1]

        events = self.event2voxelgrid(rect_x, rect_y, event_p, event_t)
        events = events.numpy()
        return events

    def h5Reader_mvsec(self, path):
        event_data = h5py.File(path, 'r')
        event_t = np.array(event_data['events/t'])
        event_x = np.array(event_data['events/x'])
        event_y = np.array(event_data['events/y'])
        event_p = np.array(event_data['events/p'])

        x_rect_map = np.loadtxt(path[:-41] + 'indoor_flying_right_x_map.txt').astype(np.float32)
        y_rect_map = np.loadtxt(path[:-41] + 'indoor_flying_right_y_map.txt').astype(np.float32)

        rect_x = x_rect_map[event_y, event_x]
        rect_y = y_rect_map[event_y, event_x]

        events = self.event2voxelgrid(rect_x, rect_y, event_p, event_t)
        events = events.numpy()
        return events
    
    def h5Reader_steic(self, path):
        event_data = h5py.File(path, 'r')
        event_t = np.array(event_data['events/t'])
        event_x = np.array(event_data['events/x'])
        event_y = np.array(event_data['events/y'])
        event_p = np.array(event_data['events/p'])

        rect_map = np.load(path[:-8]+'rect_map.npy')
        rect_xy = self.rectifyEvents(event_x, event_y, rect_map)
        rect_x = rect_xy[:, 0]
        rect_y = rect_xy[:, 1]
        
        events = self.event2voxelgrid(rect_x, rect_y, event_p, event_t)
        events = events.numpy()
        return events