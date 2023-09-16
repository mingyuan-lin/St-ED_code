import os
import argparse
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import StDataLoader
from modules.dblrnet import EDNet
from modules.dispnet import StNet
from utilities.warper import back_warp_disp
from utilities.plot_utils import show_img, ensure_dir, save_disp

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def tester_sted(dblrnet, dispnet, data_loader_test, experiment):
    tbar = tqdm(data_loader_test)
    for idx, (blurry, evgs_06, bag_name, image_iter) in enumerate(tbar):
        """ -------------------- load to GPU -------------------- """
        blurry = blurry.cuda()
        evgs_06 = evgs_06.cuda()

        disps_0 = dispnet(blurry, evgs_06)
        
        warped_evgs, _ = back_warp_disp(evgs_06, disps_0)

        pred_sharps = dblrnet(blurry, warped_evgs)

        ensure_dir(os.path.join('./test', experiment, bag_name[0], 'blur'))
        cv2.imwrite(os.path.join('./test', experiment, bag_name[0], 'blur', str(image_iter.numpy()[0]).rjust(5, '0') + '.png'), show_img(blurry))

        ensure_dir(os.path.join('./test', experiment, bag_name[0], 'pred'))
        for i in range(0, len(pred_sharps)):
            pred_name = os.path.join('./test', experiment, bag_name[0], 'pred', str(image_iter.numpy()[0]).rjust(5, '0') + '_' + str(i) + '.png')
            cv2.imwrite(pred_name, show_img(pred_sharps[i]))
        
        ensure_dir(os.path.join('./test', experiment, bag_name[0], 'disps'))
        disp_name = os.path.join('./test', experiment, bag_name[0], 'disps', str(image_iter.numpy()[0]).rjust(5, '0') + '.png')
        cv2.imwrite(disp_name, save_disp(disps_0))

    return 0


@torch.no_grad()
def main(args):
    experiment = 'dsec'

    """ -------------------- build StEDNet -------------------- """
    dispnet = StNet(baseline=args.baseline)
    dispnet = nn.DataParallel(dispnet).cuda()
    dispnet.load_state_dict(torch.load("./run/" + experiment + "/model_disp.pth"))
    print("Load pretrained dispnet from: " + "./run/" + experiment + "/model_disp.pth")
    dispnet = dispnet.eval()

    dblrnet = EDNet(baseline=args.baseline/20.)
    dblrnet = nn.DataParallel(dblrnet).cuda()
    dblrnet.load_state_dict(torch.load("./run/" + experiment + "/model_dblr.pth"))
    print("Load pretrained dblrnet from: " + "./run/" + experiment + "/model_dblr.pth")
    dblrnet = dblrnet.eval()

    """ -------------------- load dataset -------------------- """
    dataset_test = StDataLoader(args, "eval")
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    tester_sted(dblrnet, dispnet, data_loader_test, experiment)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train St-EDNet")
    parser.add_argument("--dataset_directory", type=str, default="../datasets", help="data path")
    parser.add_argument("--model_path", type=str, default="./run/", help="model saving path")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--crop_height", type=int, help="input image height", default=480)
    parser.add_argument("--crop_width", type=int, help="input image width", default=640)
    parser.add_argument('--baseline', default=80, type=float)  # dsec:80 mvsec:40 steic:60
    parser.add_argument("--description", type=str, default='')

    args = parser.parse_args()
    main(args)
