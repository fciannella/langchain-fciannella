# langchain-nvidia-trt

The `langchain-nvidia-trt` package contains LangChain integrations for connecting to LLMs served by the NVIDIA Triton+TRT-LLM setup. 

Below is an example on how to use some common functionality surrounding text-completion.

## Setting up Triton+TRT-LLM 

In this section we will provide a quickstart on how to setup a supported LLM on Triton+TRT-LLM. Please use [Optimizing Inference on Large Language Models with NVIDIA TensorRT-LLM](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/) and the [TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend) repo for more detailed instructions.

You will need a node with one or more GPUs to be able to run TRT-LLM. Depending on the size of your model and the GPU model available you will need one or more GPUs.

In this example we will set up a model on an NVIDIA H100 GPU with 80GB of memory. 

```shell
> nvidia-smi
Tue Dec 19 23:23:33 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.05    Driver Version: 525.85.05    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA H100 PCIe    On   | 00000000:C1:00.0 Off |                    0 |
| N/A   31C    P0    45W / 310W |      0MiB / 81559MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Compiling TRT-LLM

Before we deploy the model we create the TRT-LLM engines for a model we pick.

We start by compiling TRT-LLM:

```shell
apt-get update
apt-get install git-lfs

git lfs install
git clone -b release/0.5.0 https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
make -C docker release_build
```

The `make` command will take some time. It will actually spin a docker container inside of which trt-llm will be compiled.

### Getting the model 

We need to choose a model we want to test on. For instance Llama 7B:

```shell
git lfs install
git clone https://<username>:<token>@huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

For the above command you need to get an access token on HF and also to get approval from Meta (You can do this from the HF page directly). Alternatively you can get a community model:

```shell
git clone https://<username>:<token>@huggingface.co/syedhuq/llama-2-7b-hf-v2
```

Your model will be in a directory named `llama-2-7b-hf-v2`

### Compiling the model

To compile the model we will run a docker container and we will compile the model inside the container:

```shell
# Launch the Tensorrt-LLM container
make -C docker release_run LOCAL_USER=1
 
# Log in to huggingface-cli
# You can get your token from huggingface.co/settings/token
huggingface-cli login --token *****
 
# Compile model
python3 examples/llama/build.py \
    --model_dir syedhuq/llama-2-7b-hf-v2 \
    --dtype float16 \
    --use_gpt_attention_plugin float16 \
    --use_gemm_plugin float16 \
    --remove_input_padding \
    --use_inflight_batching \
    --paged_kv_cache \
    --output_dir examples/llama/out
```

We now have a model engine under `examples/llama/out/`

You can now exit this container with `exit`

### Deplpying with Triton Inference Server

#### Settings for the model to run

Now you are on the host computer, not inside any container and you should be in the `TensorRT-LLM` directory.

We start by copying the model we have compiled

```shell
# After exiting the TensorRT-LLM docker container
cd ..
git clone -b release/0.5.0 https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
cp ../TensorRT-LLM/examples/llama/out/*   all_models/inflight_batcher_llm/tensorrt_llm/1/
```

Some settings definitions

```shell
python3 tools/fill_template.py --in_place all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt decoupled_mode:true,engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1,max_tokens_in_paged_kv_cache:,batch_scheduler_policy:guaranteed_completion,kv_cache_free_gpu_mem_fraction:0.2,max_num_sequences:4

python3 tools/fill_template.py --in_place all_models/inflight_batcher_llm/preprocessing/config.pbtxt tokenizer_type:llama,tokenizer_dir:syedhuq/llama-2-7b-hf-v2

python3 tools/fill_template.py --in_place all_models/inflight_batcher_llm/postprocessing/config.pbtxt tokenizer_type:llama,tokenizer_dir:syedhuq/llama-2-7b-hf-v2
```

#### Deploying the container

We can now run the triton inference server container

```shell
docker run -it --rm --gpus all --network host --shm-size=1g -v `pwd`/all_models:/all_models -v `pwd`/scripts:/opt/scripts nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3
```

Finally we can start the server inside the container:

```shell
# Log in to huggingface-cli to get tokenizer
huggingface-cli login --token *****
 
# Install python dependencies
pip install sentencepiece protobuf
 
# Launch Server
python /opt/scripts/launch_triton_server.py --model_repo /all_models/inflight_batcher_llm --world_size 1
```

Now you have the server running inside the container and serving the model on ports `8000`, `8001`, `8002`.

Here is a command to test that it is working:

```shell
curl -X POST 10.176.11.107:8000/v2/models/ensemble/generate -d  '{ "text_input": "How do I count to nine in French?", "parameters": { "max_tokens": 100, "bad_words":[""], "stop_words":[""] }  }'
```
