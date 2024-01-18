# Inferflow

<h4>

![](https://img.shields.io/badge/PRs-welcome-brightgreen) 
<img src="https://img.shields.io/badge/Version-1.0-blue.svg" alt="Version">
<img src="https://img.shields.io/github/stars/inferflow/inferflow?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/inferflow/inferflow?color=red" alt="Issues">

</h4>



[__Inferflow__](https://inferflow.github.io/) is an efficient and highly configurable inference engine for large language models (LLMs).
With Inferflow, users can serve most of the common transformer models by simply modifying some lines in corresponding configuration files,
without writing a single line of source code. Further details can be found in our [technical report](https://arxiv.org/abs/2401.08294).

## Main Features
1. **Extensible and highly configurable**: A typical way of using Inferflow to serve a new model is editing a model specification file, but not adding/editing source codes. We implement in Inferflow a modular framework of atomic building-blocks and technologies, making it compositionally generalizable to new models. A new model can be served by Inferflow if the atomic building-blocks and technologies in this model have been "known" (to Inferflow).
2. **3.5-bit quantization**: Inferflow implements 2-bit, 3-bit, 3.5-bit, 4-bit, 5-bit, 6-bit and 8-bit quantization. Among the quantization schemes, 3.5-bit quantization is a new one introduced by Inferflow.
3. **Hybrid model partition for multi-GPU inference**: Inferflow supports multi-GPU inference with three model partitioning strategies to choose from: partition-by-layer (pipeline parallelism), partition-by-tensor (tensor parallelism), and hybrid partitioning (hybrid parallelism). Hybrid partitioning is seldom supported by other inference engines.
4. **Wide file format support** (and safely loading pickle data): Inferflow supports loading models of multiple file formats directly, without reliance on an external converter. Supported formats include pickle, safetensors, llama.cpp gguf, etc. It is known that there are security issues to read pickle files using Python codes. By implementing a simplified pickle parser in C++, Inferflow supports safely loading models from pickle data.
5. **Wide network type support**: Supporting three types transformer models: decoder-only models, encoder-only models, and encoder-decoder models.
6. **GPU/CPU hybrid inference**: Supporting GPU-only, CPU-only, and GPU/CPU hybrid inference.


Below is a comparison between Inferflow and some other inference engines:

| Model                                                        | New Model Support    | Supported File Formats   | Network Structures | Quantization Bits | Hybrid Parallelism for Multi-GPU Inference | Programming Languages |
|--------------------------------------------------------------|----------------------|--------------------------|--------------------|-------------------|:------------------------------------------:|-----------------------|
| [Huggingface Transformers](https://huggingface.co/docs/transformers/index) | Adding/editing source codes | pickle (unsafe), safetensors  | decoder-only, encoder-decoder, encoder-only | 4b, 8b | &#10008; | Python | 
| [vLLM](https://github.com/vllm-project/vllm)                 | Adding/editing source codes | pickle (unsafe), safetensors | decoder-only | 4b, 8b       | &#10008; | Python            |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)       | Adding/editing source codes |                              | decoder-only, encoder-decoder, encoder-only | 4b, 8b | &#10008; | C++, Python           |
| [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)  | Adding/editing source codes | pickle (unsafe), safetensors | decoder-only | -            | &#10008; | Python            |
| [llama.cpp](https://github.com/ggerganov/llama.cpp)          | Adding/editing source codes | gguf                         | decoder-only | 2b, 3b, 4b, 5b, 6b, 8b | &#10008; | C/C++   |
| [llama2.c](https://github.com/karpathy/llama2.c)             | Adding/editing source codes | llama2.c                     | decoder-only &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | - | &#10008; | C |
| [LMDeploy](https://github.com/InternLM/lmdeploy)             | Adding/editing source codes | pickle (unsafe), TurboMind   | decoder-only | 4b, 8b       | &#10008; | C++, Python       |
| **Inferflow**                          | **Editing configuration files** | pickle (**safe**), safetensors, gguf, llama2.c | decoder-only, encoder-decoder, encoder-only | 2b, 3b, **3.5b**, 4b, 5b, 6b, 8b | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#10004;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | C++ |

## Support Matrix

### Supported Model File Formats

- [X] Pickle (Inferflow reduces the security issue of most other inference engines in loading pickle-format files).
- [X] Safetensors
- [X] llama.cpp gguf
- [X] llama2.c

### Supported Technologies, Modules, and Options

* Supported modules and technologies related to model definition:
  - Normalization functions: STD, RMS
  - Activation functions: RELU, GELU, SILU
  - Position embeddings: ALIBI, RoPE, Sinusoidal
  - Grouped-query attention
  - Parallel attention

* Supported technologies and options related to serving:
  - Linear quantization of weights and KV cache elements: 2-bit, 3b, 3.5b, 4b, 5b, 6b, 8b
  - The option of moving part of all of the KV cache from VRAM to regular RAM
  - The option of placing the input embedding tensor(s) to regular RAM
  - Model partitioning strategies for multi-GPU inference: partition-by-layer, partition-by-tensor, hybrid partitioning
  - Dynamic batching
  - Decoding strategies: Greedy, top-k, top-p, FSD, typical, mirostat...

### Supported Transformer Models
* Decoder-only: Inferflow supports many types of decoder-only transformer models.
* Encoder-decoder: Some types of encoder-decoder models are supported.
* Encoder-only: Some types of encoder-only models are supported.

### Models with Predefined Specification Files

Users can serve a model with Inferflow by editing a model specification file. We have built [predefined specification files](data/models/) for some popular or representative models. Below is a list of such models.
- [X] Aquila (aquila_chat2_34b)
- [X] Baichuan (baichuan2_7b_chat, baichuan2_13b_chat)
- [X] BERT (bert-base-multilingual-cased)
- [X] Bloom (bloomz_3b)
- [X] Facebook m2m100 (facebook_m2m100_418m)
- [X] Falcon (falcon_7b_instruct, falcon_40b_instruct)
- [X] Internlm (internlm-chat-20b)
- [X] LLAMA2 (llama2_7b, llama2_7b_chat, llama2_13b_chat)
- [X] Mistral (mistral_7b_instruct)
- [X] Open LLAMA (open_llama_3b)
- [X] Phi-2 (phi_2)
- [X] YI (yi_6b, yi_34b_chat)


## Getting Started

### Get the Code

```bash
git clone https://github.com/inferflow/inferflow
cd inferflow
```

### Build
* Build with cmake on Linux, Mac, and WSL (Windows Subsystem for Linux):
  - Build the GPU version (that supports GPU/CPU hybrid inference):

    ```bash
    mkdir build/gpu
    cd build/gpu
    cmake ../.. -DUSE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=75
    make install -j 8
    ```

  - Build the CPU-only version:
  
    ```bash
    mkdir build/cpu
    cd build/cpu
    cmake ../.. -DUSE_CUDA=0
    make install -j 8
    ```

  Upon a successful build, executables are generated and copied to
        ```bin/release/```

* Build with Visual Studio on Windows:

  Open one of the following sln files in build/vs_projects:
  - __inferflow.sln__: The GPU version that supports GPU/CPU hybrid inference
  - __inferflow_cpu.sln__: The CPU-only version

  For the Debug configuration, executables are generated to ```bin/x64_Debug/```; while the output directory for the Release configuration is ```bin/x64_Release/```.
  
### Run the Service and Tools

* Example-1: Load a tiny model and perform inference

  Step-1: Download the model

  ```
  cd {inferflow-root-dir}/data/models/llama2.c/
  bash download.sh
  ```

  Step-2: Run the **llm_inference** tool:

  ```
  cd {inferflow-root-dir}/bin/
  release/llm_inference llm_inference.tiny.ini
  ```

* Example-2: Run the **llm_inference** tool to load a larger model for inference

  Step-1: Edit configuration file **bin/llm_inference.ini** to choose a model

  Step-2: Download the selected model
  ```
  cd {inferflow-root-dir}/data/models/{model-name}/
  bash download.sh
  ```

  Step-3: Run the tool:

  ```
  cd {inferflow-root-dir}/bin/
  release/llm_inference
  ```

* Start the Inferflow service:

  Step-1: Edit the service configuration file (bin/inferflow_service.ini)

  Step-2: Start the service:
  ```
  cd bin/release (on Windows: cd bin/x64_Release)
  ./inferflow_service
  ```

* Run the Inferflow client (for interacting with the Inferflow service via HTTP protocol to get inference results):

  Step-1: Edit the configuration file (bin/inferflow_client.ini)

  Step-2: Run the client tool:
  ```
  cd bin/release (on Windows: cd bin/x64_Release)
  ./inferflow_client
  ```

### Acknowledgements
Inferflow is inspired by the awesome projects of [llama.cpp](https://github.com/ggerganov/llama.cpp) and [llama2.c](https://github.com/karpathy/llama2.c). The CPU inference part of Inferflow is based on the [ggml](https://github.com/ggerganov/ggml) library. The FP16 data type in the CPU-only version of Inferflow is from the [Half-precision floating-point library](https://half.sourceforge.net/). We express our sincere gratitude to the maintainers and implementers of these source codes and tools.
