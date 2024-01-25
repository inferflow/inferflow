## Serving 34B or 40B models on a single 24GB-VRAM GPU
This document contains information about serving a 34B or 40B model with Inferflow on a single 24GB-VRAM GPU (e.g., RTX 3090 or 4090).

The contents in this document are suitable for the readers who have successfully built executables from the Inferflow source codes and have run a few examples.
Please refer to [**getting started**](../../#getting-started) for instructions about how to build and run the Inferflow service and tools.

### General Steps

In running the Inferflow service, a configuration file (in ini format) can be specified.
The serving configuration of a few 34B/40B models has been included in ```bin/inferflow_service.34b40b_on_24gb_vram.ini```.
The models are configured to take less than 24GB of video memory in inference time.

General steps of serving a model:

* **Step-1**: Edit configuration file ```inferflow_service.34b40b_on_24gb_vram.ini```
Please refer to [model_serving_config.md](model_serving_config.md) for instructions about editing the service configuration.

* **Step-2**: Run the service:
  ```
    #> inferflow_service inferflow_service.34b40b_on_24gb_vram.ini
  ```

General steps of inference testing with the llm_inference tool:

* **Step-1**: Edit configuration file ```inferflow_service.34b40b_on_24gb_vram.ini```
* **Step-2**: Edit configuration file ```llm_inference.34b40b_on_24gb_vram.ini```
* **Step-3**: Run llm_inference

  ```
    #> llm_inference llm_inference.34b40b_on_24gb_vram.ini
  ```

### Example-1: Serving Falcon-40B-Instruct

To serve the Falcon-40B-Instruct model, the following changes should be made for ```inferflow_service.34b40b_on_24gb_vram.ini```:
1. In section "transformer_engine", set the value of "models" to "falcon_40b_instruct".
2. In section "model.falcon_40b_instruct", set the values of the items as below:

  ```
    [model.falcon_40b_instruct]
    model_dir = ${global_model_dir}/${model_name}/
    model_specification_file = ${data_root_dir}../models/${model_name}/model_spec.json

    device_weight_data_type = Q3H
    host_kv_cache_percent = 100
    be_host_embeddings = true

    decoding_strategy = sample.top_p
    prompt_template = {query}
  ```

In the above configuration, the line "```device_weight_data_type = Q3H```" means 3.5-bit quantization is adopted for tensor weights.
Less video RAM (VRAM) is consumed with 3.5-bit quantization than 8-bit and 4-bit quantization.
To further reduce VRAM consumption, the KV cache is placed in regular RAM instead of VRAM.
This is achieved by "```host_kv_cache_percent = 100```", where "**100**" means 100 percent.
The line "```be_host_embeddings = true```" indicates that the token embedding tensor is placed in regular RAM but not VRAM.

All the three lines mentioned above are combined together to control the VRAM consumption of the model.
Please refer to [model_serving_config.md](model_serving_config.md) for more information about editing the service configuration.

The inference speed of serving Falcon-40B-Instruct on a single RTX 4090 is about 8 tokens/sec (batch-size = 1).

### Example-2: Serving Aquila_Chat2_34B

To serve the Aquila_Chat2_34B model, the following changes should be made for ```inferflow_service.34b40b_on_24gb_vram.ini```:
1. In section "transformer_engine", set the value of "models" to "falcon_40b_instruct".
2. In section "model.aquila_chat2_34b", set the values of the items as below:

  ```
    [model.aquila_chat2_34b]
    model_dir = ${global_model_dir}/${model_name}/
    model_specification_file = ${data_root_dir}../models/${model_name}/model_spec.json

    device_weight_data_type = Q3H
    device_kv_cache_data_type = Q8
    host_kv_cache_percent = 50
    be_host_embeddings = true

    decoding_strategy = {"name":"sample.top_p", "top_p":0.9, "eos_bypassing_max":0}
    prompt_template = {query}
  ```

In the above configuration, the line "```device_weight_data_type = Q3H```" means 3.5-bit quantization is adopted for tensor weights.
Less video RAM (VRAM) is consumed with 3.5-bit quantization than 8-bit and 4-bit quantization.
To further reduce VRAM consumption, part of the KV cache is placed in regular RAM instead of VRAM.
This is achieved by "```host_kv_cache_percent = 50```", where "**50**" means that 50 percent of the KV cache is in regular RAM and another 50 percent is in VRAM.
The line "```be_host_embeddings = true```" indicates that the token embedding tensor is placed in regular RAM but not VRAM.

All the three lines mentioned above are combined together to control the VRAM consumption of the model.
Please refer to [model_serving_config.md](model_serving_config.md) for more information about editing the service configuration.

The inference speed of serving Aquila_Chat2_34B on a single RTX 4090 is about 12 tokens/sec (batch-size = 1).

### Example-3: Serving Yi_34B_Chat

To serve the Yi_34B_Chat model, the following changes should be made for ```inferflow_service.34b40b_on_24gb_vram.ini```:
1. In section "transformer_engine", set the value of "models" to "falcon_40b_instruct".
2. In section "model.yi_34b_chat", set the values of the items as below:

  ```
    [model.yi_34b_chat]
    model_dir = ${global_model_dir}/${model_name}/
    model_specification_file = ${data_root_dir}../models/${model_name}/model_spec.json

    device_weight_data_type = Q3H
    device_kv_cache_data_type = Q8
    host_kv_cache_percent = 50
    be_host_embeddings = true

    max_context_len = 4096

    decoding_strategy = {"name":"sample.top_p", "top_p":0.9, "eos_bypassing_max":0}
    prompt_template = <|im_start|>system{\n}{system_prompt}<|im_end|>{\n}<|im_start|>user{\n}{query}<|im_end|>{\n}<|im_start|>assistant{\n}
  ```

In the above configuration, the line "```device_weight_data_type = Q3H```" means 3.5-bit quantization is adopted for tensor weights.
Less video RAM (VRAM) is consumed with 3.5-bit quantization than 8-bit and 4-bit quantization.
The line "```device_kv_cache_data_type = Q8```" indicates that 8-bit quantization is applied to the data items in the KV cache.
The default value of "device_kv_cache_data_type" is Q8. Therefore, this line can be removed.
To further reduce VRAM consumption, part of the KV cache is placed in regular RAM instead of VRAM.
This is achieved by "```host_kv_cache_percent = 50```", where "**50**" means that 50 percent of the KV cache is in regular RAM and another 50 percent is in VRAM.
The line "```be_host_embeddings = true```" indicates that the token embedding tensor is placed in regular RAM but not VRAM.

All the four lines mentioned above are combined together to control the VRAM consumption of the model.
Please refer to [model_serving_config.md](model_serving_config.md) for more information about editing the service configuration.

The inference speed of serving Yi_34B_Chat on a single RTX 4090 is about 12 tokens/sec (batch-size = 1).
