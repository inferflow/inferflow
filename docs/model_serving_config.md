# Model Serving Configuration



### Inference options of a model:

| Field | Remarks |
|:---------|:---------|
| **device_weight_data_type** | **Meaning**: Data type of the tensor weights stored in VRAM. <br>**Values**: F32, F16, Q8, Q5, Q4, Q3H, Q3, Q2;<br>**Default value**: F16 |
| **device_kv_cache_data_type** | **Meaning**: Data type of the KV cache elements used for GPU inference.<br>**Choices**: F32, F16, Q8;<br>**Default value**: F16 |
| **host_weight_data_type** | **Meaning**: Data type of the tensor weights stored in regular RAM. <br>**Values**: F32, F16, Q8, Q5, Q4, Q3H, Q3, Q2;<br>**Default value**: F16 |
| **host_kv_cache_data_type** | **Meaning**: Data type of the KV cache elements used for CPU inference. <br>**Values**: F32, F16, Q8;<br>**Default value**: F16 |
| **tensor_quant_threshold** | **Meaning**: The threshold of tensor size (in terms of element count) below which the tensor weights will NOT be quantized.<br>**Default value**: 2000*2000 |
| **delta_tensor_ratio** | **Meaning**: The ratio of outlier weights being selected to form a delta-tensor, stored in a sparse matrix format. <br>**Range**: [0, 0.01];<br>**Default value**: 0. |
| **host_kv_cache_percent** | **Meaning**: The percentage of KV cache data stored in regular RAM instead of VRAM.<br>**Comments**: The VRAM consumption can be reduced by placing some or all KV cache data to regular RAM, at the cost of inference efficiency. <br>**Default value**: 0 |
| **be_host_embeddings** | **Meaning**: A flag to indicate whether the embedding tensor is placed in regular RAM (but not in VRAM).<br>**Values**: true, false;<br>**Default value**: true |
| **devices** | **Meaning**: The GPU devices used in serving the model, and the partition of model data among the devices.<br>**Comments**: In the case of hybrid partition, a model is first partitioned by layer among devices groups. Then for each device group, the specific model layers are partitioned by tensor among the devices in the group. <br>**Example-1**: 0 (using device-0 to host the model) <br>**Example-2**: 0;1 (partition the model by layer among device 0 and device 1) <br>**Example-3**: 0&1 (partition the model by tensor among device 0 and device 1) <br>**Example-4**: 0&1;2&3 (partition the model by layer among device groups (0, 1) and (2, 3), then by tensor among the devices inside a device group) <br>**Default value**: (empty) |
| **max_context_len** | **Meaning**: Maximal context length in serving this model.<br>**Default value**: 0 (In this case, the training context length is used. If the training context length is not available, the value of 1024 is adopted.) |
| **decoding_strategy** | **Meaning**: The default decoding strategy in serving this model. <br>**Comments**: <br>**Values**: greedy, top_k, top_p, fsd, random_fsd, min_p, tfs, typical, mirostat <br>**Default value**: top_p |
| **encoder_input_template** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | **Meaning**: The template of the encoder input. <br>**Default value**: (Empty) |
| **decoder_input_template**<br>(or **prompt_template**) | **Meaning**: The template of the decoder input. <br>**Example-1**: {bos}{query}{res_prefix} <br>**Example-2**: {user_token}{query}{assistant_token}{res_prefix} |
