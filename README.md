# Bodies at Rest - v1.0
## 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data

<p align="center">
  <img width="98%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/intro_overview_v4.JPG?raw=true" alt="None"/>
</p>

### Paper:
Clever, Henry M., Zackory Erickson, Ariel Kapusta, Greg Turk, C. Karen Liu, Charles C. Kemp. "Bodies at Rest: 3D Human Pose and Shape Estimation from a Pressure Image using Synthetic Data." In <i>IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 2020.


### Video:
https://www.youtube.com/watch?v=UHBqw0BYWEw

### Code version:
This version is the original CVPR submission. It's old/deprecated and I suggest using v1.1 instead.

### Getting started:
PressurePose synthetic dataset: Make a new folder, `~/data_BR/synth`, and put the 6 folders (42 files) from the link below in there. For a quick start up, only download the two files in the `quick_test` folder, which is 3K images instead of 184K. 
`https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IAPI0X`

PressurePose real dataset: Make a new folder, `~/data_BR/real`, and put the 20 folders (20 participants, 60 files) from the link below in there. For a quick start up, only download the three files in the `S103.zip` folder, which represents a single participant.
`https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KOA4ML`

Clone this repository into your `~/git/` folder to get started with inspecting PressurePose and training PressureNet.

<p align="center">
  <img width="110%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/git_break1.JPG?raw=true" alt="None"/>
</p>

## What code is in here?

This repository: 

* Allows you to visualize both synthetic and real data in the PressurePose dataset. The synthetic dataset includes 206,000 fully labeled pressure images, meaning that each pressure image has a corresponding SMPL human mesh parameterized by body shape (10 PCA parameters), pose (69 joint angles + 3 DOF global rotation + 3 DOF global translation ), gender, height, and weight. The real dataset includes 1051 pressure images with co-registered point cloud data, RGB data, gender, height, and weight. 
* Has the code for PressureNet. Below we describe step-by-step how to train PressureNet. 

<p align="center">
  <img width="110%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/git_break2.JPG?raw=true" alt="None"/>
</p>

## PressurePose dataset visualization
First, install the linux packages listed further down the page.

### Synthetic data
When you run the code `python viz_synth_cvpr_release.py`, an interactive PyRender box will pop up that visualizes the ground truth human mesh and the pressure image. It will show a mesh like the two on the left below. The second set images below show a reduced set of mesh vertices, which represent only those facing an overhead camera and which overlie the pressure mat. This reduction is useful for comparing the mesh vertices to a point cloud. Use the flag `--red` to reduce the vertices in this way. You can also segment the limbs (`--seg`), which produces an image like that on the far right below. To change the synthetic data partition you are visualizing, change the variable `TESTING_FILENAME` inside the python script. 


<p align="center">
  <img width="18%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/front_synth.png?raw=true" alt="None"/>
  <img width="14%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/side_synth.png?raw=true" alt="None"/>
  <img width="18%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/front_synth_cut.png?raw=true" alt="None"/>
  <img width="12.5%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/side_synth_cut.png?raw=true" alt="None"/>
  <img width="23%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/front_synth_seg.png?raw=true" alt="None"/>
</p>


### Real data 
When you run the code `python viz_real_cvpr_release.py` you will see two pop up boxes: one for 2D data that includes RGB, depth, and pressure; the other for a 3D point cloud and pressure image rendering in PyRender that you can flip around to inspect. Make sure you include flags to specify the participant number and the type of real pose dataset. For example, you might use `--p_idx 1 --pose_type 'prescribed'` to specify the first participant in the list and the set of 48 prescribed poses. You can use numbers 1 through 20 to specify the participant, because there are 20, and pose types of `'prescribed'` and `'p_select'`, with the latter used to specify participant selected poses. Here is what you should see when you run this code:


<p align="center">
  <img width="50%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/viz_real_2D.png?raw=true" alt="None"/>
  <img width="17%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/viz_real_3D_1.png?raw=true" alt="None"/>
  <img width="14%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/viz_real_3D_2.png?raw=true" alt="None"/>
  <img width="16%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/viz_real_3D_3.png?raw=true" alt="None"/>
</p>

All image modalities are temporally synchronized, and poses are static so that any synchronization error is negligible. The real dataset is captured with a Kinect V2 and is already calibrated, and the pressure image is spatially co-registered with RGB, depth, and point cloud data. We used the code and procedures in `github.com/code-iai/iai_kinect2/` to calibrate the Kinect modalities (RGB, depth, and point cloud), including the intrinsic camera parameters. The depth image is unfiltered and noisy while the point cloud is pre-packaged as a set of 3D coordinates that has white colors filtered out - so the only points there are ones representing the person in the bed. We calibrated the pressure mat to the camera using Tungsten cubes at placed at the corner of the bed that were visible from all modalities, and used CMA-ES to optimize the 6DOF camera tranform. See the paper for details.


<p align="center">
  <img width="110%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/git_break3.JPG?raw=true" alt="None"/>
</p>

## PressureNet training
There are 3 steps to train PressureNet as implemented in the paper.
### Step 1: 
Train network 1 for 100 epochs using loss function 1. Run the following: `python train_pressurenet.py --net 1`. You can also use the flags `--htwt`, `--calnoise`, and `--small` to include height/weight data in the network, calibration noise in the network, and train on a smaller dataset size (1/4th). Use the `--qt` flag to run a quick test on a small portion of the dataset to check for bugs, or if you only downloaded the two files in the `~/data_BR/synth/quick_test` folder. By default, the network trains with a pressure image, edge map, and contact map, where the latter two are derivitives of the pressure image. To omit them, use the `--omit_cntct_sobel` flag.

It's important to visualize things to make sure your network is training OK. So if you use the `--viz` flag a set of pressure maps pops up with joint markers projected into 2D - there are 24 of them. Green - ground truth, yellow - estimated. The smaller top right images show the input channels, with the exception of height and weight. Note from the pressure image that this body is in a lateral posture, but it has just started training, so the yellow estimated joint positions are far from the ground truth.

<p align="left">
  <img width="50%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/net1.png?raw=true" alt="None"/>
</p>

### Step 2: 
Compute a new dataset that has spatial map reconstructions from the PMR output of network 1. Run the following: `python compute_network1_spatialmaps.py`. Make sure the flags on this match the flags you trained network 1 on. This will create a copy of the existing dataset plus estimated depth maps in separate files with longer filename tags.  Make sure you have at least 10GB free.

### Step 3: 
Train network 2 for 100 epochs using loss function 2. Run the following: `python train_pressurenet.py --net 2 --pmr`. Make sure the flags on this match the flags you trained network 1 on (except `--viz`, that doesn't matter). If you do visualize, expect a box like the one below to pop up. For this example, while the ground truth is in a lateral posture, the network 1 estimate outputs a pose in a prone posture. The smaller top right images show the input channels. The bottom right channels show the output reconstructed spatial maps, as well as ground truth on the far right. Here, net 2 has just started training so the output Q_2 doesn't differ substantially from the input Q_1. Use the `--qt` flag to run a quick test on a small portion of the dataset to check for bugs, or if you only downloaded the two files in the `~/data_BR/synth/quick_test` folder. For a quick test on the downloaded data use: `python train_pressurenet.py --net 2 --pmr --qt --calnoise`. 

<p align="left">
  <img width="80%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/net2_pmr.png?raw=true" alt="None"/>
</p>

The data can take a long time to load. You can use an euler angle parameterization instead of the direction cosines in the SMPL model. Use the `--losstype 'anglesEU'` flag for that. You might have to change some file directories so that the `train_pressurenet.py` knows where to find your data.


<p align="center">
  <img width="110%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/git_break4.JPG?raw=true" alt="None"/>
</p>



## PressureNet training - loss function details

Running the code `python train_pressurenet.py --net 1` for network 1 results in the L1 loss function from the paper, where the subscript `1` on the estimate denotes an estimate from network 1:

<p align="left">
  <img width="53%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/loss_1.JPG?raw=true" alt="None"/>
</p>

While the paper does not present ablative studies to check for the effect of changing the loss function, we present some flags that can generate variants. You can omit the global rotation using `--omit_root`. We recommend using a loss on the global rotation, which helps the network to get started: after the first few epochs, it becomes a very small fraction of the total loss. However, without it, we empirically found that some joint angles were likely to get 'stuck' at joint angle limits, requiring a restart. However, here it is:
<p align="left">
  <img width="32%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/loss_1_no_root.JPG?raw=true" alt="None"/>
</p>

You can optionally use PMR on Network 1 with the flag `--pmr`. We recommend using PMR only for Network 2, because it is more geared to fine-tuning the network. Using `--pmr` for network 1 will result in the following loss function:
<p align="left">
  <img width="80%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/loss_1_pmr.JPG?raw=true" alt="None"/>
</p>
We note that the ground truth reconstructed pressure map `Q-` and contact map `Co`, are quite similar yet not equal to the input pressure image `P` and contact map `Ci`. Both `Q-` and `Co` were pre-computed using a single PMR forward pass with ground truth angles from the synthetic data, and included in the released dataset. We omit the hover map `Q+` in the loss function because (1) it represents parts of the human that have no contact information and (2) unlike the reconstructed pressure map, it does not have a smooth spatial profile. However, we still use it as input in Network 2 because it comes for free with pressure map reconstruction, requires insignificant extra computation, and contains some information.

The loss on network 2 is almost identical to the loss of network 1 with PMR - the only difference is it is computed from the output of Network 2, denoted below by the appropriate subscripts on the estimates:
<p align="left">
  <img width="80%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/loss_2.JPG?raw=true" alt="None"/>
</p>

### Normalization
Besides normalizing the terms by variance, we also normalize by the number of parts within a term: i.e. N=24 joints positions to a SMPL model, or T=1728 taxels in a pressure image. We use this because it corrects for the specificity of a particular human model or a pressure image size. There are 6 parts in the first term for global rotation, because we parameterize by `atan2`, which takes as input two arguments for each DOF. We note that the shape loss is cut by an additional factor of 2. The shape is implicitly learned from joint positions, yet having some supervision on it guides the network to learn body shape better.

<p align="center">
  <img width="110%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/git_break5.JPG?raw=true" alt="None"/>
</p>


## PressureNet evaluation
First, you'll generate results files for each participant. Make a new folder `~/data_BR/final_results`. Then run `python evaluate_real.py` and specify a real data type in the PressurePose dataset with either `--pose_type 'p_select'` or `--pose_type 'prescribed'`. You should also use matching flags as before to specify inclusion of height/weight (`--htwt`), size of the dataset (`--small`), and inclusion of calibration noise (`--calnoise`). You can optionally select among the participants to evaluate using `--p_idx` followed by a number between `1` and `20`. The
 default setting is to evaluate all the participants in order from 1 to 20. You can also visualize the evaluation for a particular participant, pressure image, point cloud, and estimate in 2D or 3D using the option `--viz '2D'` or `--viz '3D'`. Note that if you visualize in 2D the results won't be saved because some of them are performed in the 3D rendering library. After you've created results files for each participant, run `python results_summary.py` specifying flags (or not) for height/weight, dataset size, and calibration noise.
 
 
<p align="center">
  <img width="110%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/git_break6.JPG?raw=true" alt="None"/>
</p>


## What other packages do I need?
* SMPL: A Skinned Multi-Person Linear Model - https://smpl.is.tue.mpg.de/. You'll have to sign up with an account to get this but it's quick. Unzip it in `~/git/`.
* PyRender - https://github.com/mmatl/pyrender
* Trimesh - https://github.com/mikedh/trimesh
* PyTorch - https://pytorch.org/
* PyTorch HMR - https://github.com/MandyMo/pytorch_HMR. Clone this and put it in `~/git/`.
* OpenCV - `pip install python-opencv`
* Open3D - `pip install open3d-python`
* LateX packages - `apt-get install dvipng texlive-fonts-recommended texlive-fonts-extra`
* Matplotlib, PyGlet, some others .... 


<p align="center">
  <img width="110%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/git_break7.JPG?raw=true" alt="None"/>
</p>

## Computer requirements
To train all 184K images, you'll need at least 64GB/CPU and 8GB/GPU ram with the present implementation. You can run smaller sizes (e.g. 32K images) if your machine has insufficient RAM. 

<p align="center">
  <img width="110%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/git_break8.JPG?raw=true" alt="None"/>
</p>

## Dataset Generation
We will release this code upon publication.

<p align="center">
  <img width="110%" src="https://github.com/Healthcare-Robotics/bodies-at-rest/blob/CVPR_2020/docs/figures/git_break9.JPG?raw=true" alt="None"/>
</p>
