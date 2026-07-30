set -e

# Note: The following commands were tested with CUDA 12.1.

conda create -n T-Search3D python=3.10
conda activate T-Search3D
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install ninja==1.10.2.3
pip install pytorch-lightning fire imageio tqdm wandb
pip install python-dotenv==0.21.0 pyviz3d==0.2.32 scipy==1.9.3 plyfile==0.7.4 scikit-learn==1.2.0 trimesh==3.17.1 loguru==0.6.0 albumentations==1.3.0 volumentations==0.1.8
pip install antlr4-python3-runtime==4.8 black==21.4b2 omegaconf==2.2.0 hydra-core==1.0.5 --no-deps

# pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps
pip install 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02' --no-deps

conda install -y openblas-devel -c anaconda

# Install MinkowskiEngine
# First check your environment
python -c "import sys; import torch; print('Python version:', sys.version); print('Torch version:', torch.__version__); print('Torch CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available());" && gcc --version | head -n 1 | cut -d' ' -f3
pip install --upgrade setuptools==59.8.0 # setuptools version must be lower than 60.0
## Option 1
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --config-settings="--blas_include_dirs=${CONDA_PREFIX}/include" --config-settings="--blas=openblas"
## Option 2
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
TORCH_CUDA_ARCH_LIST="YOUR_GPUs_CC+PTX" python setup.py install --blas=openblas --blas_include_dirs=${CONDA_PREFIX}/include --force_cuda
## If the compilation fails, follow Step 5 in this document and recompile
## https://github.com/Julie-tang00/Common-envs-issues/blob/main/Cuda12-MinkowskiEngine
cd ..

pip install pynvml==11.4.1 gpustat==1.0.0 tabulate==0.9.0 pytest==7.2.0 tensorboardx==2.5.1 yapf==0.32.0 termcolor==2.1.1 addict==2.4.0 blessed==1.19.1
pip install gorilla-core==0.2.7.8
pip install matplotlib==3.7.2
pip install cython

pip install pycocotools==2.0.6
pip install h5py==3.7.0
pip install transforms3d==0.4.1
pip install open3d==0.16.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

pip install fvcore cloudpickle Pillow

cd openmask3d/class_agnostic_mask_computation/third_party/pointnet2 && pip install .

pip install git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16 --no-deps
pip install  git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588 --no-deps

pip install ftfy regex

cd third_party/openmask3d && pip install .

cd T-FunS3D && pip install -e .

# Only for the current version of T-Search3D
pip install spacy==3.7.2
pip install numba
git clone https://github.com/Karbo123/segmentator.git && cd segmentator
cd csrc && mkdir build && cd build
cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` 
# If encounter this error "Target "cmTC_f97fd" requires the language dialect "CUDA17" (with compiler
# extensions), but CMake does not know the compile flags to use to enable it.", 
# please refer to https://stackoverflow.com/questions/61540127/set-cxx-standard-to-c17-when-combining-c-and-cuda-in-cmakelists.

# Modifying the CMakeLists.txt file as follows:
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE) 
make && make install # after install, please do not delete this folder (as we only create a symbolic link)


# Install Flash Attention 2 for pytorch 2.1.0 cuda 11.8
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install GLIP
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
pip install nltk inflect
cd GLIP && pip install -e .