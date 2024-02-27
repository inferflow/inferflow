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

## Quick Links
1. Getting started ([on Windows](docs/getting_started.win.md) | [on Linux, Mac, and Windows Subsystem for Linux (WSL)](#getting-started))
2. [Serving 34B or 40B models on a single 24GB-VRAM GPU](docs/34b40b_models_on_24gb_vram.md) (e.g., RTX 3090 and 4090)

## Milestones
* 2024-2-18: Added support for mixture-of-experts (MoE) models.
* 2024-1-17: Version 0.1.0 was formally released.

## Main Features
1. **Extensible and highly configurable**: A typical way of using Inferflow to serve a new model is editing a model specification file, but not adding/editing source codes. We implement in Inferflow a modular framework of atomic building-blocks and technologies, making it compositionally generalizable to new models. A new model can be served by Inferflow if the atomic building-blocks and technologies in this model have been "known" (to Inferflow).
2. **3.5-bit quantization**: Inferflow implements 2-bit, 3-bit, 3.5-bit, 4-bit, 5-bit, 6-bit and 8-bit quantization. Among the quantization schemes, 3.5-bit quantization is a new one introduced by Inferflow.
3. **Hybrid model partition for multi-GPU inference**: Inferflow supports multi-GPU inference with three model partitioning strategies to choose from: partition-by-layer (pipeline parallelism), partition-by-tensor (tensor parallelism), and hybrid partitioning (hybrid parallelism). Hybrid partitioning is seldom supported by other inference engines.
4. **Wide file format support** (and safely loading pickle data): Inferflow supports loading models of multiple file formats directly, without reliance on an external converter. Supported formats include pickle, safetensors, llama.cpp gguf, etc. It is known that there are security issues to read pickle files using Python codes. By implementing a simplified pickle parser in C++, Inferflow supports safely loading models from pickle data.
5. **Wide network type support**: Supporting three types transformer models: decoder-only models, encoder-only models, and encoder-decoder models.
6. **GPU/CPU hybrid inference**: Supporting GPU-only, CPU-only, and GPU/CPU hybrid inference.


Below is a comparison between Inferflow and some other inference engines:

| Inference Engine                                             | New Model Support    | Supported File Formats   | Network Structures | Quantization Bits | Hybrid Parallelism for Multi-GPU Inference | Programming Languages |
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
- [X] Deepseek (deepseek_moe_16b_base)
- [X] Facebook m2m100 (facebook_m2m100_418m)
- [X] Falcon (falcon_7b_instruct, falcon_40b_instruct)
- [X] FuseLLM (fusellm_7b)
- [X] Gemma (gemma_2b_it)
- [X] Internlm (internlm-chat-20b)
- [X] LLAMA2 (llama2_7b, llama2_7b_chat, llama2_13b_chat)
- [X] MiniCPM (minicpm_2b_dpo_bf16)
- [X] Mistral (mistral_7b_instruct)
- [X] Mixtral (mixtral_8x7b_instruct_v0.1)
- [X] Open LLAMA (open_llama_3b)
- [X] OPT (opt_350m, opt_13b, opt_iml_max_30b)
- [X] Orion (orion_14b_chat)
- [X] Phi-2 (phi_2)
- [X] Qwen (qwen1.5_7b_chat) 
- [X] XVERSE (xverse_13b_chat)
- [X] YI (yi_6b, yi_34b_chat)


## Getting Started

**Windows** users: Please refer to [docs/getting_started.win.md](docs/getting_started.win.md) for the instructions about building and running the Inferflow tools and service on Windows.

The following instructions are for **Linux**, **Mac**, and **WSL** (Windows Subsystem for Linux).

### Get the Code

```bash
git clone https://github.com/inferflow/inferflow
cd inferflow
```

### Build
* Build the GPU version (that supports GPU/CPU hybrid inference):

  ```bash
  mkdir build/gpu
  cd build/gpu
  cmake ../.. -DUSE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=75
  make install -j 8
  ```

* Build the CPU-only version:
  
  ```bash
  mkdir build/cpu
  cd build/cpu
  cmake ../.. -DUSE_CUDA=0
  make install -j 8
  ```

Upon a successful build, executables are generated and copied to
        ```bin/release/```

### Run the LLM Inferencing Tool (bin/llm_inference)

* **Example-1**: Load a tiny model and perform inference

  - **Step-1**: Download the model

    ```
    #> cd {inferflow-root-dir}/data/models/llama2.c/
    #> bash download.sh
    ```
    Instead of running the above batch script, you can also manually download the model files and copy them to the above folder. The source URL and file names can be found from download.sh.

  - **Step-2**: Run the **llm_inference** tool:

    ```
    #> cd {inferflow-root-dir}/bin/
    #> release/llm_inference llm_inference.tiny.ini
    ```
    Please note that it is okay for ```llm_inference``` and ```llm_inference.tiny.ini``` not being in the same folder (llm_inference.tiny.ini is in bin/ and llm_inference is in bin/release/).

* **Example-2**: Run the **llm_inference** tool to load a larger model for inference

  - **Step-1**: Edit configuration file **bin/inferflow_service.ini** to choose a model.

    In the "transformer_engine" section of bin/inferflow_service.ini, there are multiple lines starting with "```models = ```" or "```;models = ```".
    The lines starting with the "**;**" character are comments.
    To choose a model for inference, please uncomment the line corresponding to this model, and comment the lines of other models.
    By default, the **phi-2** model is selected.
    Please refer to [docs/model_serving_config.md](docs/model_serving_config.md) for more information about editing the configuration of inferflow_service.
  
  - **Step-2**: Download the selected model
    ```
    #> cd {inferflow-root-dir}/data/models/{model-name}/
    #> bash download.sh
    ```

  - **Step-3**: Edit configuration file **bin/llm_inference.ini** to choose or edit a query.

    In the configuration file, queries are organized into query lists. A query list can contain one or multiple queries.
    Different query lists are for different purposes. For example, ```query_list.decoder_only``` is for testing decoder-only models. Its detailed information can be configured in the ```query_list.decoder_only``` section.
    The starting line of this section is "```query_count = 1```", which means only one query is included in this query list.
    Among the following lines with key ```query1```, only one line is uncommented and therefore effective, whereas other lines (i.e., the lines starting with a "**;**" character) are commented.
    You can choose a query for testing by uncommenting this query and commenting all the other queries. You can, of course, add new queries or change the content of an existing query.
  
  - **Step-4**: Run the tool:

    ```
    #> cd {inferflow-root-dir}/bin/
    #> release/llm_inference
    ```

### Run the Inferflow Service (bin/inferflow_service)

  * **Step-1**: Edit the service configuration file (bin/inferflow_service.ini)

  * **Step-2**: Start the service:
    ```
    #> cd bin
    #> release/inferflow_service
    ```

### Test the Inferflow service

Run an HTTP client, to interact with the Inferflow service via the HTTP protocol to get inference results.

* **Option-1**. Run the Inferflow client tool: inferflow_client

  - **Step-1**: Edit the configuration file (bin/inferflow_client.ini) to set the service address, query text, and options.

  - **Step-2**: Run the client tool to get inference results.
  ```
  #> cd bin
  #> release/inferflow_client
  ```

* **Option-2** The CURL command

  You can also use the CURL command to send a HTTP POST request to the Inferflow service and get inference results. Below is an example:
  ```
  curl -X POST -d '{"text": "Write an article about the weather of Seattle.", "res_prefix": "", "decoding_alg": "sample.top_p", "random_seed": 1, "temperature": 0.7, "is_streaming_mode": false}' localhost:8080
  ```

* **Option-3**. Use GUI REST client (e.g., the Chrome extension of ```Tabbed Postman```).

  - **URL**: ```http://localhost:8080``` (If you access the service from a different machine, please replace "localhost" with the service IP)
  
  - **HTTP method**: ```POST```
  
  - **Example body text**: ```{"text": "Write an article about the weather of Seattle.", "res_prefix": "", "decoding_alg": "sample.top_p", "random_seed": 1, "temperature": 0.7, "is_streaming_mode": 0}```



## Reference
If you are interested in our work, please kindly cite:
```bib
@misc{shi2024inferflow,
    title={Inferflow: an Efficient and Highly Configurable Inference Engine for Large Language Models},
    author={Shuming Shi and Enbo Zhao and Deng Cai and Leyang Cui and Xinting Huang and Huayang Li},
    year={2024},
    eprint={2401.08294},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Acknowledgements
Inferflow is inspired by the awesome projects of [llama.cpp](https://github.com/ggerganov/llama.cpp) and [llama2.c](https://github.com/karpathy/llama2.c). The CPU inference part of Inferflow is based on the [ggml](https://github.com/ggerganov/ggml) library. The FP16 data type in the CPU-only version of Inferflow is from the [Half-precision floating-point library](https://half.sourceforge.net/). We express our sincere gratitude to the maintainers and implementers of these source codes and tools.
