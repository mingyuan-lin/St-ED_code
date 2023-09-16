# Learning Parallax for Stereo Event-based Motion Deblurring
## [Paper]() | [Website](https://mingyuan-lin.github.io/St-ED_web/)

Due to the extremely low latency, events have been recently exploited to supplement lost information for motion deblurring. Existing approaches largely rely on the perfect pixel-wise alignment between intensity images and events, which is not always fulfilled in the real world. To tackle this problem, we propose a novel coarse-to-fine framework, named NETwork of Event-based motion Deblurring with STereo event-intensity cameras (\myname), to recover high-quality images directly from the misaligned inputs, consisting of a single blurry image and the concurrent event streams. Specifically, the coarse spatial alignment of the blurry image and the event streams is first implemented with a cross-modal stereo matching module without the need for ground-truth depths. Then, a dual-feature embedding architecture is proposed to gradually build the fine bidirectional association of the coarsely aligned data and reconstruct the sequence of the latent sharp images. Furthermore, we build a new dataset with STereo Event and Intensity Cameras (\mydata), containing real-world events, intensity images, and dense disparity maps. Experiments on real-world datasets demonstrate the superiority of the proposed network over state-of-the-art methods.

## Environment setup
- Python 3.7.0
- Pytorch 1.9.1
- NVIDIA GPU + CUDA 11.1
- numpy, argparse, tqdm, natsort, opencv, h5py, hdf5plugin

## Download data
In our paper, we build a real-world dataset **StEIC** which contains intensity images, dense disparities and real-world events. (The data is coming soon.)

## Quick start
### Test
```bash
python test.py
```

## Citation
If you find our work useful in your research, please cite:

```
@article{lin2023learning,
        title={Learning Parallax for Stereo Event-based Motion Deblurring},
        author={Lin, Mingyuan and Chu, He and Yu, Lei},
        journal={arXiv},
        year={2023}
        }
```