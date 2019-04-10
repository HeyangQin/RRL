#!/bin/bash

if [ -z $TENSORFLOW_SERVING_REPO_PATH ]; then
    TENSORFLOW_SERVING_REPO_PATH="/home/ResearchProjects/serving_1_0"
fi
INITIAL_PATH=$(pwd)
export TF_NEED_CUDA=1
export TF_NEED_GCP=1
export TF_NEED_JEMALLOC=1
export TF_NEED_HDFS=0
export TF_NEED_OPENCL=0
export TF_ENABLE_XLA=0
export TF_CUDA_VERSION=9.0
export TF_CUDNN_VERSION=7
export TF_CUDA_COMPUTE_CAPABILITIES="3.5,5.2"
CUDA_PATH="/usr/local/cuda"
if ! [ -d CUDA_PATH ]; then
#    CUDA_PATH="/opt/cuda"
	echo "CUDA PATH $CUDA_PATH"
fi
export CUDA_TOOLKIT_PATH=$CUDA_PATH
export CUDNN_INSTALL_PATH=$CUDA_PATH
export GCC_HOST_COMPILER_PATH=$(which gcc-5 || which gcc)
export PYTHON_BIN_PATH=$(which python)
export CC_OPT_FLAGS="-march=native"

function python_path {
  "$PYTHON_BIN_PATH" - <<END
from __future__ import print_function
import site
import os

try:
  input = raw_input
except NameError:
  pass

python_paths = []
if os.getenv('PYTHONPATH') is not None:
  python_paths = os.getenv('PYTHONPATH').split(':')
try:
  library_paths = site.getsitepackages()
except AttributeError:
 from distutils.sysconfig import get_python_lib
 library_paths = [get_python_lib()]
all_paths = set(python_paths + library_paths)

paths = []
for path in all_paths:
  if os.path.isdir(path):
    paths.append(path)

if len(paths) == 1:
  print(paths[0])
else:
  ret_paths = ",".join(paths)
  print(ret_paths)
END
}

export PYTHON_LIB_PATH=$(python_path)

#cd $TENSORFLOW_SERVING_REPO_PATH
cd tensorflow
./configure
cd ..
sed -i.bak 's/@org_tensorflow\/\/third_party\/gpus\/crosstool/@local_config_cuda\/\/crosstool:toolchain/g' tools/bazel.rc
if [ -e $(which gcc-5) ]; then
    sed -i.bak 's/"gcc"/"gcc-5"/g' tensorflow/third_party/gpus/cuda_configure.bzl
fi
bazel build -c opt --config=cuda --spawn_strategy=standalone //tensorflow_serving/model_servers:tensorflow_model_server

cd $INITIAL_PATH
