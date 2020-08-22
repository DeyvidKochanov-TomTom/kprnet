# [KPRNet: Improving projection-based LiDARsemantic segmentation](https://arxiv.org/pdf/2007.12668.pdf)

![Video](kprnet.gif)

## Installation

Install [apex](https://github.com/NVIDIA/apex) also the python only build. Then:

```bash
pip install -r requirements.txt
```

## Experiment 

Download pre-trained [resnext_cityscapes_2p.pth](https://drive.google.com/file/d/1aioKjoxcrfqUtkWQgbo64w8YoLcVAW2Z/view?usp=sharing). The path should be given in `model_dir`.  CityScapes pretraining will be added later.

The result from paper is trained on 8 16GB GPUs (total batch size 24).

To train run:

```bash
python train_kitti.py --semantic-kitti-dir path_to_semantic_kitti --model-dir location_where_your_pretrained_model_is --checkpoint-dir your_output_dir
```

## Acknowledgments
[KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) 
[RangeNet++](https://github.com/PRBonn/lidar-bonnetal) 
[HRNet](https://github.com/HRNet)

## Reference

KPRNet appears in ECCV workshop Perception for Autonomous Driving.

```
@article{kochanov2020kprnet,
  title={KPRNet: Improving projection-based LiDAR semantic segmentation},
  author={Kochanov, Deyvid and Nejadasl, Fatemeh Karimi and Booij, Olaf},
  journal={arXiv preprint arXiv:2007.12668},
  year={2020}
}
```
