## Codebase

The codebase includes parts of the real-world demonstration of Hestia-NBV: Hestia-NBV for next-best-view prediction and MASt3R for RGB-to-depth conversion. The codebase is based on the NVIDIA IsaacLab 2.1 (July 12, 2025), MASt3R, dust3r_MAD3D, and Hestia-NBV repo. Please follow [Hestia-NBV](https://github.com/johnnylu305/Hestia-NBV) to setup IsaacLab environment. For the real-world drone communication and control component, please refer to the [repository](TBD) organized by our robotics scientists, Da Xiao and Trung Le.


## Codebase

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

### Install Hestia-NBV-Demo

```
# clone this repo
git clone --branch new_isaac_demo --recursive https://github.com/johnnylu305/mast3r.git
# install required packages
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
pip install colorama
pip install numpy==1.26

# You need to modify the following path in sim.py accordingly:
# sys.path.append("/home/dsr/Documents/mad3d/demo/isaac-sim-4.5.0/home/Hestia-NBV")
# sys.path.append("/home/dsr/Documents/mad3d/demo/isaac-sim-4.5.0/home/Hestia-NBV/source/isaaclab_tasks/isaaclab_tasks/direct/single_drone")
# sys.path.append("/home/dsr/Documents/mad3d/demo/isaac-sim-4.5.0/home/Hestia-NBV/scripts/mad3d")
# img_root, mode_name

# You need to modify scripts/mad3d/sb3_policy_cus.py in Hestia-NBV for real-world demo:
# prob_grid[env, :, :, :5] = 0.5
# if masked_distances.min() >= 1e6:
# new_world_actions[0][2] = 0.5

# You can run demo with placeholder data in the example folder
python3 sim.py
# for 3D scene reconstruction
python3 demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
```

### Hestia-NBV-Demo

```
# You can run demo with placeholder data in the example folder
python3 sim.py
# for 3D scene reconstruction
python3 demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

# To run the real-world demo:
# 1. Empty the example folder and set up the real-world drone codebase.
# 2. Run sim.py for next-best-view prediction and RGB-to-depth conversion.
# 3. Run the real-world drone codebase to send commands, receive data, and control the drone.
```

## Citation

If you find the codebase useful for your research, please consider citing:

```
@misc{lu2025hestiahierarchicalnextbestviewexploration,
      title={Hestia: Hierarchical Next-Best-View Exploration for Systematic Intelligent Autonomous Data Collection}, 
      author={Cheng-You Lu and Zhuoli Zhuang and Nguyen Thanh Trung Le and Da Xiao and Yu-Cheng Chang and Thomas Do and Srinath Sridhar and Chin-teng Lin},
      year={2025},
      eprint={2508.01014},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2508.01014}, 
}
```

Please also consider citing NVIDIA IsaacLab, and MASt3R:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

```
@misc{mast3r_eccv24,
      title={Grounding Image Matching in 3D with MASt3R}, 
      author={Vincent Leroy and Yohann Cabon and Jerome Revaud},
      booktitle = {ECCV},
      year = {2024}
}
```

```
@inproceedings{dust3r_cvpr24,
      title={DUSt3R: Geometric 3D Vision Made Easy}, 
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      booktitle = {CVPR},
      year = {2024}
}
```


