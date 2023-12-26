<div align="center">
# Inferflow
<div>
  <a href='https://scholar.google.com/citations?user=Lg31AKMAAAAJ&hl=en/' target='_blank'>Shuming Shi</b></a>&emsp;
  <a target='_blank'>Enbo Zhao </a>&emsp;
</div>
<div>Tencent AI Lab</div>

<div>
<h4>

![](https://img.shields.io/badge/PRs-welcome-brightgreen) 
<img src="https://img.shields.io/badge/Version-1.0-blue.svg" alt="Version">
<img src="https://img.shields.io/github/stars/HillZhang1999/ICD?color=yellow" alt="Stars">
<img src="https://img.shields.io/github/issues/HillZhang1999/ICD?color=red" alt="Issues">

</h4>
</div>

</div>


__Inferflow__ is an efficient and highly configurable inference engine and service for large language models (LLMs).
With Inferflow, users can serve most of the common transformer models by simply modifying some lines in corresponding configuration files,
without writing a single line of source code.

## Main features


1. Providing support for commonly used inference optimization techniques, including: Efficient tensor operation kernels, operator fusion, KV caching, dynamic batching, and speculative decoding.
2. Multiple strategies for reducing VRAM consumption: quantization, allowing for placing some layers to CPU memory, the choice of placing the KV cache and the embedding tensor(s) to CPU memory.
3. Multi-GPU inference with a variety of model partitioning strategies to choose from: partition-by-layer, partition-by-tensor, and __hybrid partitioning__.
4. GPU/CPU hybrid inference
5. Quantization support: 2-bit, 2.5-bit, 3-bit, 3.5-bit, 4-bit, 5-bit, 6-bit and 8-bit quantization.
6. Providing support for loading models of multiple file formats directly, without reliance on an external converter. Supported formats include pickle, safetensors, llama.cpp gguf, etc.
7. Supporting serving most of the common transformer models shared on the web, including various kinds of decoder-only models (e.g., LLaMA/LLaMA2 and Bloom), encoder-only models, and encoder-decoder models.
Many key modules of the model network can be specified by configration, including layer normalization functions, activation functions, position embedding algorithms, tensor names, etc.
8. Allowing for loading multiple models in one engine/service and choosing one on the fly during inference time.

**Supported model file formats:**

- [X] Pickle (Inferflow reduces the security issue of most other inference engines in loading pickle-format files).
- [X] Safetensors
- [X] llama.cpp gguf
- [X] llama2.c

**Supported models:**

Users can flexibly configure models based on [custom configuration files] (docs/model_setup.md). Here are some official config models provided.
- [X] LLaMA and LLaMA2 series and their variants (e.g., Mistral, falcon, baichuan)
- [X] Bloom
- [X] BERT
- [X] GPT/GPT2/GPT3
- [X] T5



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
        {inferflow-root-dir}/bin/release/

* Build with Visual Studio on Windows:

  Open one of the following .sln files in build/vs_projects:
  - __inferflow.sln__: The GPU version that supports GPU/CPU hybrid inference
  - __inferflow_cpu.sln__: The CPU-only version

