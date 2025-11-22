# DSFNet-based-SEGO
DSFNet is a 3D SSC framework for indoor RGB-D. DAM uses depth completion to repair noisy/incomplete depths. SSFM fuses 2D semantic priors with geometric features for accurate voxel occupancy and labels. The resulting maps drive SEGO, a risk-aware planner producing smoother, safer paths. Validated on NYU/NYUCAD and Gazebo.

![](./figure/SYO.png)



![](./figure/DSFNet.png)

## Contents

0. [Installation](#installation)
0. [Data Preparation](#Data-Preparation)
0. [Train and Test](#Train-and-Test)
0. [Visualization and Evaluation](#visualization-and-evaluation)
0. [Citation](#Citation)

## Installation

### Environment

- Ubuntu 20.04
- python 3.6
- CUDA 11.8

### Requirements:

- [pytorch](https://pytorch.org/)≥2.4.0
- [torch_scatter](https://github.com/rusty1s/pytorch_scatter)
- imageio
- scipy
- scikit-learn
- tqdm

You can install the requirements by running `pip install -r requirements.txt`.

If you use other versions of PyTorch or CUDA, be sure to select the corresponding version of torch_scatter.


## Data Preparation

### Download dataset

The raw data can be found in [SSCNet](https://github.com/shurans/sscnet).

The repackaged data can be downloaded via 
[Google Drive](https://drive.google.com/drive/folders/15vFzZQL2eLu6AKSAcCbIyaA9n1cQi3PO?usp=sharing)
or
[BaiduYun(Access code:lpmk)](https://pan.baidu.com/s/1mtdAEdHYTwS4j8QjptISBg).

The repackaged data includes:

```python
rgb_tensor   = npz_file['rgb']		# pytorch tensor of color image
depth_tensor = npz_file['depth']	# pytorch tensor of depth 
tsdf_hr      = npz_file['tsdf_hr']  	# flipped TSDF, (240, 144, 240)
tsdf_lr      = npz_file['tsdf_lr']  	# flipped TSDF, ( 60,  36,  60)
target_hr    = npz_file['target_hr']	# ground truth, (240, 144, 240)
target_lr    = npz_file['target_lr']	# ground truth, ( 60,  36,  60)
position     = npz_file['position']	# 2D-3D projection mapping index
```

### 

## Train 

### Configure the data path in [config.py](https://github.com/waterljwant/SSC/blob/master/config.py#L9)

```
'train': '/path/to/your/training/data'

'val': '/path/to/your/testing/data'
```

### Train

Edit the training script [run_train.sh](https://github.com/XH-hlx/DSFNet-based-Indoor-Nvigation/blob/main/DSFNet/run_train.sh#L4), then run

```
bash run_train.sh
```

## Visualization and Evaluation

comging soon


## Citation

If you find this work useful in your research, please cite our paper(s):

    

}
