## Overview
[![YT screenshot](screenshot.png)](https://www.youtube.com/watch?v=CTAcCmC0gYk)

This is a fork of [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) with changes for the weighted flow objective proposed in our [V2V](https://github.com/swami1995/V2V) paper (Supplementary section). Please refer to the original repository for setup and evaluation instructions.

To start training,
```
python train.py --datapath=datasets/TartanAir --gpus=4 --lr=0.00025 --name=wt_floss --num_workers=2 --w3=3
```
