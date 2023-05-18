### Set up the python environment

```
conda create -n neuralbody python=3.7
conda activate neuralbody

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 10.0, install torch 1.4 built from cuda 10.0
pip install torch==1.4.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

```

### Set up datasets

#### People-Snapshot dataset

1. Download the People-Snapshot dataset [here](https://graphics.tu-bs.de/people-snapshot).
2. Process the People-Snapshot dataset using the [script](https://github.com/zju3dv/neuralbody#process-people-snapshot).
3. Create a soft link:
    ```
    ROOT=/path/to/neuralbody
    cd $ROOT/data
    ln -s /path/to/people_snapshot people_snapshot
    ```

#### ZJU-Mocap dataset

1. If someone wants to download the ZJU-Mocap dataset, please fill in the [agreement](https://pengsida.net/project_page_assets/files/ZJU-MoCap_Agreement.pdf), and email me (pengsida@zju.edu.cn) and cc Xiaowei Zhou (xwzhou@zju.edu.cn) to request the download link.
2. Create a soft link:
    ```
    ROOT=/path/to/neuralbody
    cd $ROOT/data
    ln -s /path/to/zju_mocap zju_mocap
    ```
