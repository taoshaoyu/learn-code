# Text Generation
We provide the inference benchmarking script `run_generation.py` for large language models text generation.<br/>
Support most of large language models, such as GPT-J, LLaMA, OPT, BLOOM and etc.<br/>
And script `run_generation_with_deepspeed.py` for distributed with DeepSpeed.<br/>
And script `run_model_int8.py` for int8.<br/>

## Setup
```bash
WORK_DIR=$PWD
# GCC 11 is required, please set it firstly
#     If no gcc11, recommended: CentOS gcc-toolset, Ubuntu update-alternatives
# Create environment (conda recommended)
conda create -n llm python=3.9 -y
conda install cmake ninja mkl mkl-include -y
conda install gperftools -c conda-forge -y

# Install PyTorch
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
# Install IPEX with semi-compiler, require gcc 11+
git clone --branch llvmorg-13.0.0 https://github.com/llvm/llvm-project
cd llvm-project && mkdir build && cd build
cmake ../llvm -DCMAKE_INSTALL_PREFIX=${PWD}/_install/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_ENABLE_TERMINFO=OFF -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
make install -j$(nproc)
# make sure llvm-config-13 in your PATH
ln -s ${PWD}/_install/llvm/bin/llvm-config ${CONDA_PREFIX}/bin/llvm-config-13
cd ../../
git clone --branch llm_feature_branch https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu
cd frameworks.ai.pytorch.ipex-cpu
git submodule sync && git submodule update --init --recursive
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
python setup.py install
cd ../
# Install transformers
pip install git+https://github.com/intel-sandbox/transformers.git@llm_4.28.1
# Install others deps
pip install cpuid accelerate datasets sentencepiece protobuf==3.20.3

# Setup environment variables for performance on Xeon
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

# [Optional] Neural compressor only for int8 case
git clone https://github.com/intel/neural-compressor.git
cd neural-compressor
pip install -r requirements.txt
python setup.py install
cd ../

# [Optional] The following is only for DeepSpeed case
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ccl.git
cd frameworks.ai.pytorch.torch-ccl
git checkout public_master
git submodule sync
git submodule update --init --recursive
python setup.py install
cd ../
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
python -m pip install -r requirements/requirements.txt
python setup.py install
cd ../
git clone https://github.com/oneapi-src/oneCCL.git
cd oneCCL
mkdir build
cd build
cmake ..
make -j install
source _install/env/setvars.sh
cd ../..
```

## Performance
```bash
# bfloat16 and float32
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <physical cores list> python run_generation.py --device cpu -m EleutherAI/gpt-j-6b --dtype bfloat16 --ipex --jit

# INT8
INT8_ARGS=" --int8_bf16_mixed " # for mixed bfloat16, mixed float32 uses --int8
# GPT-J quantization
python run_gptj_int8.py --quantize --inc_smooth_quant --lambada --output_dir "saved_results" --jit ${INT8_ARGS}
# LLaMA quantization
python run_llama_int8.py --ipex_smooth_quant --lambada --output_dir "saved_results" --jit ${INT8_ARGS}
# GPT-NEOX quantization
python run_gptx_int8.py --ipex_smooth_quant --lambada --output_dir "saved_results" --jit ${INT8_ARGS}
# benchmark
OMP_NUM_THREADS=<physical cores num> numactl -m <node N> -C <cpu list> python run_<MODEL>_int8.py --quantized_model_path "./saved_results/best_model.pt" --benchmark --jit ${INT8_ARGS}
```

## Performance with DeepSpeed
```bash
threads_per_core=$(lscpu |grep 'Thread(s) per core:' |sed 's/[^0-9]//g')
cores_per_node=$(numactl -H |grep "node 0 cpus:" |sed 's/.*://' |awk -v tpc=$threads_per_core '{print int(NF / tpc)}')
export OMP_NUM_THREADS=${cores_per_node}
unset KMP_AFFINITY

# auto TP
deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark --device cpu -m EleutherAI/gpt-j-6b --dtype bfloat16 --ipex --jit
# kernel inject True
deepspeed --bind_cores_to_rank run_generation_with_deepspeed.py --benchmark --device cpu -m EleutherAI/gpt-j-6b --dtype bfloat16 --ipex --jit --ki
```
