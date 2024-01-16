# Model Setup

## Instructions about editing model_spec.json


#### Fields of the network_structure object:

| Field | Remarks |
|:---------|:---------|
| type | Network structure type, which can be one of the following values: <br>&nbsp;&nbsp;transformer <br>&nbsp;&nbsp;&nbsp;&nbsp;transformer.decoder_only <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;transformer.llama <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;transformer.bloom <br>&nbsp;&nbsp;&nbsp;&nbsp;transformer.encoder_only <br>&nbsp;&nbsp;&nbsp;&nbsp;transformer.encoder_decoder |
| normalization_function |  |
| activation_function | The activation function used in the model. The following functions are supported in Inferflow so far, <br>&nbsp;&nbsp;relu <br>&nbsp;&nbsp;gelu <br>&nbsp;&nbsp;silu |
| position_embedding | The position embedding algorithm used in the model. The following algorithms are supported in Inferflow so far, <br>&nbsp;&nbsp;alibi <br>&nbsp;&nbsp;rope <br>&nbsp;&nbsp;sinusoidal |
| qk_column_order |  |
| qkv_format |  |
| normalize_lm_head | Possible values: true, false <br>Whether to normalize the lm_head (or output.weight) tensor before performing any inference operations. |
| tensor_name_prefix | The prefix of the tensor names |
| tensor_name_mapping | Its value is a JSON object containing the mapping from the tensor names in the model file to standard tensor names. |

&nbsp;

#### Standard tensor names for a Transformer model:

| Standard Name | Remarks |
|:-----------|:------------|
| enc.token_embeddings.weight | The token embeddings tensor of the encoder |
| enc.{i}.self_attn.pre_norm.weight/.bias | The weight and bias for the normalization operation before a self-attention layer of the encoder |
| enc.{i}.self_attn.qkv.weight/.bias | Tensor qkv.weight is the concatenation of wq.weight, wk.weight, and wv.weight. Vector qkv.bias is the concatenation of wq.bias, wk.bias, and wv.bias. |
| enc.{i}.self_attn.wq.weight/.bias | WQ and its corresponding bias vector in a self-attention layer of the encoder |
| enc.{i}.self_attn.wk.weight/.bias | WK and its corresponding bias vector in a self-attention layer of the encoder |
| enc.{i}.self_attn.wv.weight/.bias | WV and its corresponding bias vector in a self-attention layer of the encoder |
| enc.{i}.self_attn.wo.weight/.bias | WO and its corresponding bias vector in a self-attention layer of the encoder |
| enc.{i}.feed_forward.pre_norm.weight/.bias | The weight and bias for the normalization operation before a feed-forward layer of the encoder |
| enc.{i}.feed_forward.w1.weight/.bias | The weight and bias of the gate projection layer in in a feed-forward layer of the encoder. |
| enc.{i}.feed_forward.w2.weight/.bias | The weight and bias of the down projection layer in in a feed-forward layer of the encoder. |
| enc.{i}.feed_forward.w3.weight/.bias | The weight and bias of the up projection layer in in a feed-forward layer of the encoder. |
| enc.{i}.feed_forward.post_norm.weight/.bias | The weight and bias for the normalization operation after a feed-forward layer of the encoder |
| enc.output_norm.weight/.bias | The weight and bias for the normalization operation performed on the overall output of the encoder |
| dec.token_embeddings.weight | The token embeddings tensor of the decoder |
| dec.{i}.self_attn.pre_norm.weight/.bias | The weight and bias for the normalization operation before a self-attention layer of the decoder |
| dec.{i}.self_attn.qkv.weight/.bias | Tensor qkv.weight is the concatenation of wq.weight, wk.weight, and wv.weight. Vector qkv.bias is the concatenation of wq.bias, wk.bias, and wv.bias. |
| dec.{i}.self_attn.wq.weight/.bias | WQ and its corresponding bias vector in a self-attention layer of the decoder |
| dec.{i}.self_attn.wk.weight/.bias | WK and its corresponding bias vector in a self-attention layer of the decoder |
| dec.{i}.self_attn.wv.weight/.bias | WV and its corresponding bias vector in a self-attention layer of the decoder |
| dec.{i}.self_attn.wo.weight/.bias | WO and its corresponding bias vector in a self-attention layer of the decoder |
| dec.{i}.cross_attn.pre_norm.weight/.bias | The weight and bias for the normalization operation before a cross-attention layer of the decoder |
| dec.{i}.cross_attn.qkv.weight/.bias | Tensor qkv.weight is the concatenation of wq.weight, wk.weight, and wv.weight. Vector qkv.bias is the concatenation of wq.bias, wk.bias, and wv.bias. |
| dec.{i}.cross_attn.wq.weight/.bias | WQ and its corresponding bias vector in a cross-attention layer of the decoder |
| dec.{i}.cross_attn.wk.weight/.bias | WK and its corresponding bias vector in a cross-attention layer of the decoder |
| dec.{i}.cross_attn.wv.weight/.bias | WV and its corresponding bias vector in a cross-attention layer of the decoder |
| dec.{i}.cross_attn.wo.weight/.bias | WO and its corresponding bias vector in a cross-attention layer of the decoder |
| dec.{i}.feed_forward.pre_norm.weight/.bias | The weight and bias for the normalization operation before a feed-forward layer of the decoder |
| dec.{i}.feed_forward.w1.weight/.bias | The weight and bias of the gate projection layer in in a feed-forward layer of the decoder. |
| dec.{i}.feed_forward.w2.weight/.bias | The weight and bias of the down projection layer in in a feed-forward layer of the decoder. |
| dec.{i}.feed_forward.w3.weight/.bias | The weight and bias of the up projection layer in in a feed-forward layer of the decoder. |
| dec.{i}.feed_forward.post_norm.weight/.bias | The weight and bias for the normalization operation after a feed-forward layer of the decoder |
| dec.output_norm.weight/.bias | The weight and bias for the normalization operation performed on the overall output of the decoder |
| output.weight | lm_head |
