# EVE-NeRF

[<font size="4"><u>Paper</u></font>](https://arxiv.org/abs/2311.11845) <font size="5">&#124;</font> [<font size="4"><u>Pretrained Model</u></font>](https://drive.google.com/drive/folders/1NgYdVFeUuPwyj7HPdQp5dbLR2Ktli1Et?usp=sharing)

PyTorch implementation for the CVPR 2024 paper: [<u>Entangled View-Epipolar Information Aggregation for Generalizable Neural Radiance Fields</u>](https://arxiv.org/abs/2311.11845). The paper introduces a method called EVE-NeRF for synthesizing novel views across new scenes in a generalizable manner. Unlike existing methods that consider cross-view and along-epipolar information independently, EVE-NeRF conducts view-epipolar feature aggregation in an entangled manner by incorporating appearance continuity and geometry consistency priors. The proposed approach improves the accuracy of 3D scene geometry and appearance reconstruction compared to prevailing single-dimensional aggregation methods. This repository is built based on the offical repository of [<u>IBRNet</u>](https://github.com/googleinterns/IBRNet) and [<u>GNT</u>](https://github.com/VITA-Group/GNT).


<center>
    <img src="assets/pipline.png" width="80%" />
</center>



## Installation
Clone this repository:
```Shell
git clone https://github.com/tatakai1/EVENeRF.git
cd EVENeRF/
```

Tested on **Ubuntu22.04**, **python3.8**, **cuda12.1**, **pytorch2.1.1**. Install environment:
```Shell
conda create -n evenerf python=3.8
conda activate evenerf
# You can install pytorch corresponding to your cuda version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tensorboard ConfigArgParse imageio matplotlib numpy opencv_contrib_python Pillow scipy imageio-ffmpeg lpips scikit-image==0.19.3
```

## Datasets
Please follow [<u>IBRNet</u>](https://github.com/googleinterns/IBRNet) and [<u>GNT</u>](https://github.com/VITA-Group/GNT) to download training and evaluation datasets. All datasets should unzip to the dictionary ```/data``` within the project folder. 
```
├──data/
    ├──ibrnet_collected_1/
    ├──ibrnet_collected_2/
    ├──real_iconic_noface/
    ├──spaces_dataset/
    ├──RealEstate10K-subset/
    ├──google_scanned_objects/
    ├──nerf_synthetic/
    ├──nerf_llff_data/
    ├──shiny/
```

(optianal) If you want to evaluate on dataset **Shiny**, please refer to [<u>NEX's</u>](https://github.com/nex-mpi/nex-code/) repository to download [<u>Shiny dataset</u>](https://vistec-my.sharepoint.com/:f:/g/personal/pakkapon_p_s19_vistec_ac_th/EnIUhsRVJOdNsZ_4smdhye0B8z0VlxqOR35IR3bp0uGupQ?e=TsaQgM).

## Evaluation
The evaluation process is relatively slow. It takes about 3 minutes to render an image with a resolution of 1008×756 using one single RTX4090. Therefore, in addition to using multiple gpus, you can speed up the testing process by  ```--render_stride 2``` (reducing the rendered image resolution) or  ```--testskip 16``` (increasing the interval between rendered images). 
```Shell
# LLFF datasets
python3 eval.py --config configs/eve_llff.txt --run_val --render_stride 1 --testskip 8 --ckpt_path /path/to/prtrained/model
# Synthetic datasets
python3 eval.py --config configs/eve_synthetic.txt --run_val --render_stride 1 --testskip 8 --N_samples 192 --ckpt_path /path/to/prtrained/model
# Shiny datasets
python3 eval.py --config configs/eve_shiny.txt --run_val --render_stride 1 --testskip 8 --N_samples 192 --ckpt_path /path/to/prtrained/model
```


## Training
We train the cross-scene's model with 6 V100 32G for 250,000 iterations, per iteration with 682 rays. If you can only train on a single GPU (not recommended), please adjust hyperparameters such as ```N_rand, lrate_feature, lrate_eve``` in ```configs/eve_full.txt``` according to the GPU's memory.
```Shell
# multi-gpus for cross scene
python3 -m torch.distributed.launch --nproc_per_node=6 train.py --config configs/eve_full.txt  --distributed True
# sigle gpu for cross scene
python3 train.py --config configs/eve_full.txt  --distributed False
# multi-gpus for finetuing on a specific scene
python3 -m torch.distributed.launch --nproc_per_node=6 train.py --config configs/eve_finetune.txt  --distributed True
```


## Rendering videos
You can use render.py to render videos for the real forward-facing scenes as follows (your own datasets through camera calibration and "horns" in LLFF): 

<p float="left">
  <img src="assets/desk.gif" width="47%" />
  <img src="assets/horns.gif" width="47%" /> 
</p>

```Shell
python3 render.py --config configs/eve_llff.txt --eval_dataset llff_render --eval_scenes horns --expname eve_finetune_horns
```