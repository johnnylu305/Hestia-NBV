---
<div align="center">

# <b>Hestia</b>: Hierarchical Next-Best-View Exploration for Systematic Intelligent Autonomous Data Collection

Cheng-You Lu<sup>1</sup>, Zhuoli Zhuang<sup>1</sup>, Nguyen Thanh<sup>1</sup>, Trung Le<sup>1</sup>, Da Xiao<sup>1</sup>, Yu-Cheng Chang<sup>1</sup>, Thomas Do<sup>1</sup>, Srinath Sridhar<sup>2</sup>, Chin-Teng Lin<sup>1</sup>

<p><sup>1</sup>University of Technology Sydney &nbsp;&nbsp;<sup>2</sup>Brown University &nbsp;&nbsp;

### [Projectpage](TBD) · [Paper](TBD) · [Video](TBD)

</div>

## Introduction

We propose <b>Hestia</b>, a generalizable RL-based next-best-view planner that actively predicts viewpoints for data capture in 3D reconstruction tasks.

Video TBD

## Codebase

The codebase contains the training and testing code for Hestia in a simulation environment. The codebase is based on the NVIDIA IsaacLab 2.1 (July 12, 2025) repository. Please follow the [IsaacLab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
) to install IsaacSim 4.5.0 and clone this repository as the IsaacLab repository. You can also follow our old-school steps to set up the codebase environment.

## Environment

The original Hestia codebase was built on IsaacLab 4.2.0. Due to significant updates to IsaacLab since then, we have upgraded it to IsaacLab 2.1 and IsaacSim 4.5.0, using an environment with Ubuntu 22.04, Python 3.10, and an NVIDIA RTX A6000 GPU.

### Install IsaacSim

We will install IsaacSim and IsaacLab using Singularity (a container platform). Please install [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html
) first. This is the old-school method, and we recommend following the new [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) instead, as it should be more convenient. If you choose to follow our method, please also obtain a token from [NVIDIA](org.ngc.nvidia.com/setup/api-keys) for IsaacSim Docker file authentication.

```
# authentication
export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
export SINGULARITY_DOCKER_PASSWORD=[Token]

# build IsaacSim
# ex: singularity build --sandbox isaac-sim-4.5.0/ docker://nvcr.io/nvidia/isaac-sim:4.5.0
singularity build --sandbox [container_name] [docker_link]

# validate IsaacSim
cd [container_name]/isaac-sim/
./isaac-sim.sh
```

### Install IsaacLab
```
# clone repo
cd [container_name]/home/
git clone https://github.com/johnnylu305/Hestia-NBV.git
cd Hestia-NBV/

# link IsaacSim
ln -s ../../isaac-sim/ _isaac_sim

# environment variables (optional?)
export ISAACSIM_PATH="${HOME}/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# conda environment
./isaaclab.sh --conda isaaclab2.1
conda activate isaaclab2.1

# install packages
./isaaclab.sh --install

# if any package such as open3d is missing, you can try:
./isaaclab.sh -p -m pip install [missing package]

# depending on your GPU and cuda version you may need to:
pip list | grep -i cuda
pip uninstall -y nvidia-cuda-cupti-cu11 nvidia-cuda-nvrtc-cu11 nvidia-cuda-runtime-cu11
./isaaclab.sh -p -m pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129

# validate IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task=Isaac-Cartpole-Direct-v0
```

## Training
We assume the dataset is stored under "/home/dsr/Documents/mad3d/New_Dataset20/". Please update this path in both the codebase and command line to match your local setup. If you encounter errors such as "Failed to create change watch", rebooting the system usually resolves the issue.

### Training Dataset (from scratch)
```
# download dataset (please update BASE_PATH)
./isaaclab.sh -p scripts/mad3d/download_objaverse.py
# or download the dataset [here](TBD)

# increase the VMA limit if necessary
sudo sysctl -w vm.max_map_count=524288
# glb to usd and occ
./isaaclab.sh -p scripts/mad3d/convert_mesh_to_usd_occ.py /home/dsr/Documents/mad3d/New_Dataset20/objaverse/hf-objaverse-v1/glbs/ /home/dsr/Documents/mad3d/New_Dataset20/objaverse/preprocess/

# remove invalid 3D model
./isaaclab.sh -p scripts/mad3d/remove_incomplete_data.py --root_directory /home/dsr/Documents/mad3d/New_Dataset20/objaverse/preprocess/ --mode objaverse

# preprocess occ
./isaaclab.sh -p scripts/mad3d/preprocess_occ.py /home/dsr/Documents/mad3d/New_Dataset20/objaverse/preprocess/

# select n shapes for train and m shapes for test
./isaaclab.sh -p scripts/mad3d/select_shapes.py --root_path /home/dsr/Documents/mad3d/New_Dataset20/objaverse/preprocess/ --num_shapes 30100 --train_size 30000 --test_size 100
```

### Hestia Training
```
# increase the VMA limit if necessary
sudo sysctl -w vm.max_map_count=1048576

# train model
# since we remove and load a new scene for each episode, you may encounter errors such as "PhysX error: PxRigidActor::detachShape", which can be ignored...
./isaaclab.sh -p scripts/mad3d/sb3_single_drone.py --task=MAD3D-v0 --enable_cameras --num_envs 256
```

## Testing

### Ground Truth Point Clouds
```
# generate raw pcd
# ex: ./isaaclab.sh -p scripts/mad3d/convert_mesh_to_pcd.py --filter_input /home/dsr/Documents/mad3d/New_Dataset20/objaverse/preprocess/test.txt /home/dsr/Documents/mad3d/New_Dataset20/objaverse/hf-objaverse-v1/glbs/ /home/dsr/Documents/mad3d/New_Dataset20/objaverse/PCD/
 ./isaaclab.sh -p scripts/mad3d/convert_mesh_to_pcd.py --filter_input [path_to_dataset_txt] [path_to_raw_mesh] [output_path]

# remove occluded points
# ex: ./isaaclab.sh -p scripts/mad3d/remove_non_visible_pcd.py --pointcloud /home/dsr/Documents/mad3d/New_Dataset20/objaverse/PCD/ --occ /home/dsr/Documents/mad3d/New_Dataset20/objaverse/preprocess/ --output /home/dsr/Documents/mad3d/New_Dataset20/objaverse/PCD_RF/  --mode objaverse
./isaaclab.sh -p scripts/mad3d/remove_non_visible_pcd.py --pointcloud [raw_pcd_path] --occ [hollow_occ_path] --output [save_folder_path]   --mode [dataset]
```

### Hestia Testing
```
# test Hestia on the dataset's test split using a specific object position and model checkpoint
# results are stored under "logs/sb3/MAD3D-v0/model/"
# ex: ./isaaclab.sh -p scripts/mad3d/sb3_inference.py --input /home/dsr/Documents/mad3d/New_Dataset20/objaverse/preprocess/test.txt --task=MAD3D-v0 --checkpoint logs/sb3/MAD3D-v0/model/model_4352000_steps.zip --trans 0 0 0  --enable_camera
./isaaclab.sh -p scripts/mad3d/sb3_inference.py --input [path_to_ext_file] --task=[task] --checkpoint [path_to_checkpoint] --trans [object position]  --enable_camera
```

## More datasets for testing

### Houses3K (from scratch)
```
# download raw houses3k data here (https://github.com/darylperalta/Houses3K)
# or here (TBD)
# fbx to glb
python3 scripts/mad3d/fbx_to_glb.py --root /home/dsr/Documents/mad3d/New_Dataset20/houses3k/Raw_FBX/ --output /home/dsr/Documents/mad3d/New_Dataset20/houses3k/Raw_GLB/

# glb to usd and occ
./isaaclab.sh -p scripts/mad3d/convert_mesh_to_usd_occ.py /home/dsr/Documents/mad3d/New_Dataset20/houses3k/Raw_GLB/ /home/dsr/Documents/mad3d/New_Dataset20/houses3k/preprocess/

# remove invalid 3D model
./isaaclab.sh -p scripts/mad3d/remove_incomplete_data.py --root_directory /home/dsr/Documents/mad3d/New_Dataset20/houses3k/preprocess/ --mode houses3k

# preprocess occ
./isaaclab.sh -p scripts/mad3d/preprocess_occ.py /home/dsr/Documents/mad3d/New_Dataset20/houses3k/preprocess/

# run circular path
./isaaclab.sh -p scripts/mad3d/occ_from_circular_path.py --input /home/dsr/Documents/mad3d/New_Dataset20/houses3k/preprocess/ --vis --enable_cameras

# select n shapes for train and m shapes for test
./isaaclab.sh -p scripts/mad3d/select_shapes.py --root_path /home/dsr/Documents/mad3d/New_Dataset/houses3k/preprocess/ --num_shapes 356 --train_size 256 --test_size 100

# verification
# the score is usually 0.99
./isaaclab.sh -p scripts/mad3d/compare_hollow_to_circular.py --root /home/dsr/Documents/mad3d/New_Dataset/houses3k/preprocess/

# generate raw pcd
# ex: ./isaaclab.sh -p scripts/mad3d/convert_mesh_to_pcd.py --filter_input /home/dsr/Documents/mad3d/New_Dataset20/houses3k/preprocess/test.txt /home/dsr/Documents/mad3d/New_Dataset20/houses3k/Raw_GLB/ /home/dsr/Documents/mad3d/New_Dataset20/houses3k/PCD/
./isaaclab.sh -p scripts/mad3d/convert_mesh_to_pcd.py --filter_input [path_to_dataset_txt] [path_to_raw_mesh] [output_path]

# remove occluded points
# ex: ./isaaclab.sh -p scripts/mad3d/remove_non_visible_pcd.py --pointcloud /home/dsr/Documents/mad3d/New_Dataset20/houses3k/PCD/ --occ /home/dsr/Documents/mad3d/New_Dataset20/houses3k/preprocess/ --output /home/dsr/Documents/mad3d/New_Dataset20/houses3k/PCD_RF/  --mode houses3k
./isaaclab.sh -p scripts/mad3d/remove_non_visible_pcd.py --pointcloud [raw_pcd_path] --occ [hollow_occ_path] --output [save_folder_path]   --mode [dataset]
```

### OmniObject3D (from scratch)
```
# download OmniObject3D raw data subset here (TBD)
# rename raw data (if necessary)
sh scripts/mad3d/rename_omni3d.sh

# obj to usd and occ
./isaaclab.sh -p scripts/mad3d/convert_mesh_to_usd_occ.py /home/dsr/Documents/OMNI3D/omni3d_raw_data/  /home/dsr/Documents/mad3d/New_Dataset20/omni3d/preprocess/

# remove invalid 3D model
./isaaclab.sh -p scripts/mad3d/remove_incomplete_data.py --root_directory /home/dsr/Documents/mad3d/New_Dataset20/omni3d/preprocess/ --mode omniobject3d

# preprocess occ
./isaaclab.sh -p scripts/mad3d/preprocess_occ.py /home/dsr/Documents/mad3d/New_Dataset20/omni3d/preprocess/

# select n shapes for train and m shapes for test
 ./isaaclab.sh -p scripts/mad3d/select_shapes.py --root_path /home/dsr/Documents/mad3d/New_Dataset20/omni3d/preprocess/ --num_shapes 200 --train_size 0 --test_size 200

# generate raw pcd
# ex: ./isaaclab.sh -p scripts/mad3d/convert_mesh_to_pcd.py --filter_input /home/dsr/Documents/mad3d/New_Dataset20/omni3d/preprocess/test.txt /home/dsr/Documents/OMNI3D/omni3d_raw_data/ /home/dsr/Documents/mad3d/New_Dataset20/omni3d/PCD/
./isaaclab.sh -p scripts/mad3d/convert_mesh_to_pcd.py --filter_input [path_to_dataset_txt] [path_to_raw_mesh] [output_path]

# remove occluded points
# ex: ./isaaclab.sh -p scripts/mad3d/remove_non_visible_pcd.py --pointcloud /home/dsr/Documents/mad3d/New_Dataset20/omni3d/PCD/ --occ /home/dsr/Documents/mad3d/New_Dataset20/omni3d/preprocess/ --output /home/dsr/Documents/mad3d/New_Dataset20/omni3d/PCD_RF/  --mode omniobject3d
./isaaclab.sh -p scripts/mad3d/remove_non_visible_pcd.py --pointcloud [raw_pcd_path] --occ [hollow_occ_path] --output [save_folder_path]   --mode [dataset]
```
