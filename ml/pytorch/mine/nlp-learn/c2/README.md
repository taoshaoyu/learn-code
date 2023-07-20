# Text Generation
We provide the inference benchmarking script `run_generation.py` for large language models text generation.
Support most of large language models, such as GPT-J, LLaMA, OPT, BLOOM and etc.

## Setup
```bash
WORK_DIR=$PWD
# Create environment (conda recommended)
conda create -n llm python=3.9 -y
conda install ninja mkl mkl-include -y
conda install gperftools -c conda-forge -y

# Install PyTorch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
# Install IPEX with semi-compiler, require gcc 11+
git clone --branch llvmorg-13.0.0 https://github.com/llvm/llvm-project
cd llvm-project && mkdir build && cd build
cmake ../llvm -DCMAKE_INSTALL_PREFIX=${PWD}/_install/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_ENABLE_TERMINFO=OFF -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
make install -j$(nproc)
ln -s ${PWD}/_install/llvm/bin/llvm-config /usr/local/bin/llvm-config-13
cd ../../
git clone https://github.com/intel/intel-extension-for-pytorch
cd intel-extension-for-pytorch
git submodule sync && git submodule update --init --recursive
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
python setup.py install
cd ../
# Install transformers
pip install git+https://github.com/intel-sandbox/transformers.git@llm_4.28.1
# Install others deps
pip install cpuid accelerate datasets sentencepiece protobuf==3.20.3
```

## Performance
```bash
# Setup Environment Variables for best performance on Xeon
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

# support single socket and multiple sockets
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_generation.py --device cpu -m EleutherAI/gpt-j-6b --dtype bfloat16 --ipex --jit
```
