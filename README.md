<p align="center">
  <img src="assets/t-funs3d_logo.png" alt="NOVA3R logo" width="224">
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2606.05975v1"><img src="https://img.shields.io/badge/arXiv-2603.04179-b31b1b.svg" alt="arXiv"></a>
  <a href="https://t-funs3d.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
</p>

# T-FunS3D: Task-Driven Hierarchical Open-Vocabulary 3D Functionality Segmentation

**[ICRA 2026]** The repository contains the official implementation of [T-FunS3D](https://t-funs3d.github.io/). **T-FunS3D** constructs an open-vocabulary scene graph using 3D point cloud and posed RGB-D images of an indoor environment. Once a free-form task query is assigned, **T-FunS3D** segments the functional interactive object parts in the 3D point cloud.


> **T-FunS3D: Task-Driven Hierarchical Open-Vocabulary 3D Functionality Segmentation**<br> 
> [Jingkun Feng](), [Reza Sabzevari]() <br> 
> ICRA 2026

**[[Paper](https://arxiv.org/abs/2606.05975v1)] [[Project Page](https://t-funs3d.github.io/)]**


## Requirements

- **Python**: 3.10
- **PyTorch / CUDA**: the installer pins PyTorch 2.1.2 and CUDA Toolkit 12.1.1.
- **Driver**: a working NVIDIA driver compatible with CUDA 12.1 (530.30.02 or newer recommended).
- **GPU**: NVIDIA GPU with ≥16GB VRAM.

## Installation

```bash
# Clone the repository
git clone --recursive https://github.com/EdwardjkFeng/T-FunS3D.git
cd T-FunS3D

# Automated setup (creates or reuses the Python 3.10 environment T-FunS3D)
bash install.sh
```

> [!IMPORTANT]
> The installer puts `nvcc`, CUDA headers/libraries, GCC 11, and build tools inside
> `T-FunS3D`. It does not install a system toolkit, modify your NVIDIA driver, or
> change shell startup files. A different system CUDA version is fine: the driver
> must support the pinned toolkit. The CUDA version displayed by `nvidia-smi` is
> the driver's supported version, not proof of an installed toolkit.
>
> GPU compute capabilities are detected from all devices visible to PyTorch
> (respecting `CUDA_VISIBLE_DEVICES`) and used to compile native extensions.
> CUDA 12.1 cannot support every future NVIDIA GPU: unsupported architectures
> fail with an explicit message rather than building for the wrong device.
> Your GPU also needs enough memory for the models you run.

<details> 
  <summary> Optional installation settings </summary>

  ```bash
  ENV_NAME=T-FunS3D-test MAX_JOBS=4 bash install.sh
  # FlashAttention 2 is optional and requires Ampere/Ada/Hopper GPUs.
  INSTALL_FLASH_ATTN=1 bash install.sh
  ```

  Use a dedicated environment; rerunning updates its dependencies to the pinned
  versions. Existing environments must use Python 3.10. `MAX_JOBS` defaults to 2
  to limit memory usage during compilation. Keep `third-party/segmentator`
  after installation because its Python package links to that build directory.
  The script resolves paths relative to itself, so it can also be invoked from
  another directory. It prints `Starting` and `Completed` for each numbered stage,
  and names the failing stage if a command stops. Rerun the same command after
  fixing a failure; the pinned source checkouts under `third-party` are reused.
  Use `bash install.sh --help` for settings. The last stage checks dependencies
  and runs CUDA smoke tests. Model checkpoints and datasets are downloaded
  separately.
</details>


Activate the conda env once installation is completed
```bash
conda activate T-FunS3D
```

<!-- ## Demo

Run scene graph construction and functionality segmentation on an example scene.

```bash
conda activate t-funs3d

# Open-vocabulary instance segemnetation
bash run_openmask3d_single_scene.sh

# Scene graph construction
bash run_scene_graph_construction.sh

# Task queries parser
bash run_task_parser.sh

# Functionality segmentation
bash run_functionality_segmentation.sh
```

Outputs are saved to 'demo/outputs/example_scene/' -->

## Data Preparation
### Checkpoints and example data
Download checkpoints and example data (also from the repository root) by running the download script:
```bash
bash download_data.sh
```

### SceneFun3D dataset
We download the data split of SceneFun3D using the published scripts of Fun3DU.
1. Create dataset root folder `$ROOT` (`datasets/scenefun3d/` is the default path in the scripts).
2. Download the file lists folder from the [original dataset repo](https://github.com/SceneFun3D/scenefun3d/tree/main/benchmark_file_lists) and put it in the `$ROOT`.
```bash
export ROOT="$PWD/datasets/scenefun3d"
mkdir -p "$ROOT/benchmark_file_lists"
for file in train_val_set.csv train_scenes.txt val_scenes.txt; do
    curl -fL "https://raw.githubusercontent.com/SceneFun3D/scenefun3d/main/benchmark_file_lists/$file" -o "$ROOT/benchmark_file_lists/$file"
done
```
3. Create the lists of two splits by running the following scripts:
```bash
cd data_preparation
python make_video_list.py train
python make_video_list.py val
```
4. Download the data splits: 
```bash
python sun3d/data_asset_download.py --split custom --video_id_csv $ROOT/benchmark_file_lists/val_set.csv --download_dir $ROOT/val --dataset_asset laser_scan_5mm crop_mask annotations descriptions hires_wide hires_wide_intrinsics hires_depth hires_poses

python sun3d/data_asset_download.py --split custom --video_id_csv $ROOT/benchmark_file_lists/train_set.csv --download_dir $ROOT/train --dataset_asset laser_scan_5mm crop_mask annotations descriptions hires_wide hires_wide_intrinsics hires_depth hires_poses
```

Run the following script to prepare the scenefun3d data for the pipeline. The step includes converting data structures and preprocessing the point clouds.
```bash
bash scenefun3d_batch_data_preprocess.sh <batch_id> # The script prepares the data batch-wise. One batch contains 10 scenes by default.
```



## Run T-FunS3D on SceneFun3D
Before running this script, adjust the folowings: 
1. `ROOT`: dataset root
2. `OUTPUT_DIRECTORY`, `OUTPUT_FOLDER_DIRECTORY`: output paths

```bash
bash run_t-funs3d.sh
```



## BibTeX

If you find T-FunS3D useful for your research and applications, please cite us using this BibTex:

```bibtex
@inproceedings{feng2026tfuns3d,
  author    = {Feng, Jingkun and Sabzevari, Reza},
  title     = {T-FunS3D: Task-Driven Hierarchical Open-Vocabulary 3D Functionality Segmentation},
  booktitle = {2026 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full terms. Code from Third-party (e.g., [OpenMask3D](https://github.com/OpenMask3D/openmask3d), [SceneFun3D](https://scenefun3d.github.io/documentation/), [Fun3DU](https://github.com/tev-fbk/fun3du)) retains its original license.

## Acknowledgements
We build on prior advances in open-vocabulary 3D segementation, foundation models, and vision-language models. Our codebase is implemented based on [OpenMask3D](https://github.com/OpenMask3D/openmask3d), [SceneFun3D](https://scenefun3d.github.io/documentation/), [Fun3DU](https://github.com/tev-fbk/fun3du), [FG-CLIP](https://huggingface.co/qihoo360/fg-clip-base), [QWen3](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), and [Molmo](https://github.com/allenai/molmo). We sincerely appreciate the authors for their wonderful work and for releasing their code, models, and data processing scripts.