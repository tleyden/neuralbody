# Install

## 1. I just did “pip install torch” since I have cuda 11.6 installed on this box

## 2. I got into some PDH, so I changed the requirements.txt file to:

```
open3d>=0.9.0.0
PyYAML>=5.3.1
tqdm>=4.28.1
tensorboardX>=1.2
termcolor>=1.1.0
scikit-image>=0.14.2
opencv-contrib-python>=3.4.2.17
opencv-python>=3.4.2.17,<4
imageio==2.3.0
trimesh==3.8.15
plyfile==0.6
PyMCubes==0.1.0
pyglet==1.4.0b1
chumpy
```

## 3. Compile spconv hack

I had to comment out a #include line to get spconv to compile, as mentioned here: https://github.com/traveller59/spconv/issues/464#issuecomment-1442055025 🤮

## 4a. To process the data set, it was missing a dependency: `pip install h5py`

## 4b. Missing models

To process the data set, it was missing some models.  Looks like they can be downloaded from link mentioned here: https://github.com/zju3dv/neuralbody/issues/43#issuecomment-937575241

## 5. Imageio

I had to upgrade “pip install --upgrade imageio” to get past this error: https://github.com/danijar/handout/issues/22 when trying to run the “python run.py --type visualize” command.  Now it segfaults.. which might be because I’m on a headless box, or might be for some other reason

## 6a. OpenGL

When running "tools/render_mesh.py", it was missing opengl.  `pip install PyOpenGL`. 

## 6b. Install freeglut3

I hit this same error (https://github.com/zju3dv/neuralbody/issues/94) when running “python tools/render_mesh.py --exp_name female3c --dataset people_snapshot --mesh_ind 226” and I posted my workaround on the issue.  

```
sudo apt-get install freeglut3-dev
```

Now I’m hitting “freeglut (foo): failed to open display ‘’” which is most likely due to running on a headless machine

## 7. I hit the spconv error: prepareSubMGridKernel failed

```
RuntimeError: .../spconv/src/spconv/indice.cu 274
cuda execution failed with error 3 initialization error
prepareSubMGridKernel failed
```

(same as https://github.com/traveller59/spconv/issues/262) 

and tried uninstalling my spconv and reinstalling spconv2 from pip.

## 8. Hitting error related to spconv1 vs spconv2 behavior:

```
$ python run.py --type visualize --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c vis_mesh True train.num_workers 0
load model: data/trained_model/if_nerf/female3c/latest.pth
Traceback (most recent call last):
  File "run.py", line 126, in <module>
    globals()['run_' + args.type]()
  File "run.py", line 88, in run_visualize
    epoch=cfg.test.epoch)
  File "/home/tleyden/DevLibraries/neuralbody/lib/utils/net_utils.py", line 378, in load_network
    net.load_state_dict(pretrained_model['net'], strict=strict)
  File "/opt/miniconda/miniconda3/envs/neuralbody/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1672, in load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for Network:
	size mismatch for xyzc_net.conv0.0.weight: copying a param with shape torch.Size([3, 3, 3, 16, 16]) from checkpoint, the shape in current model is torch.Size([16, 3, 3, 3, 16]).
	size mismatch for xyzc_net.conv0.3.weight: copying a param with shape torch.Size([3, 3, 3, 16, 16]) from checkpoint, the shape in current model is torch.Size([16, 3, 3, 3, 16]).
```

This issue might be relevant: https://github.com/zju3dv/neuralbody/issues/121 - it's because I installed spconv2 from pip.

## 9. Train model from scratch 

I gave up on using the pre-trained model, trying to train a model from scratch.  Hit new error:

```
$ python train_net.py --cfg_file configs/snapshot_exp/snapshot_f3c.yaml exp_name female3c resume False
make_network is loading network from:  lib/networks/latent_xyzc.py
remove contents of directory data/record/if_nerf/female3c
rm: cannot remove 'data/record/if_nerf/female3c/*': No such file or directory
Traceback (most recent call last):
  File "train_net.py", line 108, in <module>
    main()
  File "train_net.py", line 104, in main
    train(cfg, network)
  File "train_net.py", line 23, in train
    evaluator = make_evaluator(cfg)
  File "/home/tleyden/DevLibraries/neuralbody/lib/evaluators/make_evaluator.py", line 16, in make_evaluator
    return _evaluator_factory(cfg)
  File "/home/tleyden/DevLibraries/neuralbody/lib/evaluators/make_evaluator.py", line 8, in _evaluator_factory
    evaluator = imp.load_source(module, path).Evaluator()
  File "/opt/miniconda/miniconda3/envs/neuralbody/lib/python3.7/imp.py", line 171, in load_source
    module = _load(spec)
  File "<frozen importlib._bootstrap>", line 696, in _load
  File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 728, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "lib/evaluators/if_nerf.py", line 3, in <module>
    from skimage.measure import compare_ssim
ImportError: cannot import name 'compare_ssim' from 'skimage.measure' (/opt/miniconda/miniconda3/envs/neuralbody/lib/python3.7/site-packages/skimage/measure/__init__.py)
```

Looks like an api rename issue: https://github.com/williamfzc/stagesepx/issues/150

Training in progress.

# Generate EasyMoCap video

## Generate keypoints via mediapipe

```
python3 apps/preprocess/extract_keypoints.py data/1v1p_keypoint_test/1v1p/ --mode mp-holistic 
```

## Generate EasyMocap video

```
data=<path/to/data>
python3 apps/demo/mocap.py ${data} --work internet
```

# Train neuralbody on custom video

## Generate keypoints via mediapipe

```
python3 apps/preprocess/extract_keypoints.py data/1v1p_keypoint_test/1v1p/ --mode mp-holistic 
```


## Generate VideoAvatar related metadata files

The PeopleSnapshot dataset contains the following files, which are needed by NeuralBody:

According to [this ticket](https://github.com/zju3dv/neuralbody/issues/42) these files can be generated via [VideoAvatars](https://github.com/thmoa/videoavatars)

### camera.pkl

There is a script to generate this.  It wants some things like focal distance, which might be in the image metadata or available from the ios API.  They are optional however.

https://github.com/thmoa/videoavatars/blob/master/prepare_data/create_camera.py


### keypoints.hdf5

These needed to be generated by OpenPose and then converted using:

2djoints2hdf5.py - "This script stores OpenPose 2D keypoints from json files in the given directory in a compressed hdf5 file."

Alternatively, MediaPipe could be used (much easier to install) and then converted to OpenPose format using: https://github.com/Atif-Anwer/Mediapipe_to_OpenPose_JSON

### masks.hdf5

I believe this is generated from https://github.com/kmaninis/OSVOS-PyTorch or similar.  In the [VideoAvatars paper](https://arxiv.org/pdf/1803.04758.pdf) it mentions:

> Our method relies on a foreground segmentation of the images. Therefore, we adopt the CNN based video segmentation method of [9] and train it with 3-4 manual segmentations per sequence

They needed to be converted using this script:

masks2hdf5.py - "This script stores image masks from a directory in a compressed hdf5 file."


### consensus.obj + .pkl

I believe these are both generated by step2_consensus.py in the VideoAvatars repo.

### reconstructed_poses.hdf5

I think this is the output from step1_pose.py

## Generate annots.npy

```
python zju_smpl/easymocap_to_neuralbody.py --input_dir /home/tleyden/DevLibraries/EasyMocap/data/1v1p_keypoint_test/1v1p --type annots
```

After this step, you will have a new annots.npy in the input directory

## Create config file

Duplicate `monocular_custom.yaml` and customize, eg:

```
task: 'if_nerf'
gpus: [0]

parent_cfg: 'configs/snapshot_exp/snapshot_f3c.yaml'

train_dataset:
    data_root: '../EasyMocap/data/1v1p_keypoint_test/1v1p'
    human: 'custom'
    ann_file: '../EasyMocap/data/1v1p_keypoint_test/1v1p/annots.npy'
    split: 'train'

test_dataset:
    data_root: '../EasyMocap/data/1v1p_keypoint_test/1v1p'
    human: 'custom'
    ann_file: '../EasyMocap/data/1v1p_keypoint_test/1v1p/annots.npy'
    split: 'test'

# data options
ratio: 1.
training_view: [0, 6, 12, 18]
num_train_frame: 300
smpl: 'smpl'
vertices: 'vertices'
params: 'params'
big_box: True
```


## Run training

```
python train_net.py --cfg_file configs/monocular_1v1p_custom.yaml exp_name 1v1p_keypoint_test_exp
```

If it fails with `FileNotFoundError: [Errno 2] No such file or directory: '../EasyMocap/data/1v1p_keypoint_test/1v1p/camera.pkl'` you need to generate the file using the VideoAvatars repo.