1. I just did “pip install torch” since I have cuda 11.6 installed on this box

2. I got into some PDH, so I changed the requirements.txt file to:

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

3. I had to comment out a #include line to get spconv to compile, as mentioned here: https://github.com/traveller59/spconv/issues/464#issuecomment-1442055025 🤮

4a. To process the data set, it was missing a dependency: `pip install h5py`

4b. To process the data set, it was missing some models.  Looks like they can be downloaded from link mentioned here: https://github.com/zju3dv/neuralbody/issues/43#issuecomment-937575241

5. I had to upgrade “pip install --upgrade imageio” to get past this error: https://github.com/danijar/handout/issues/22 when trying to run the “python run.py --type visualize” command.  Now it segfaults.. which might be because I’m on a headless box, or might be for some other reason

6a. When running "tools/render_mesh.py", it was missing opengl.  `pip install PyOpenGL`

6b. I hit this same error (https://github.com/zju3dv/neuralbody/issues/94) when running “python tools/render_mesh.py --exp_name female3c --dataset people_snapshot --mesh_ind 226” and I posted my workaround on the issue.  Now I’m hitting “freeglut (foo): failed to open display ‘’” which is most likely due to running on a headless machine