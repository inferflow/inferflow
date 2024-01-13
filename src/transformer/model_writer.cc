#include "model_writer.h"
#include "sslib/path.h"
#include "sslib/string_util.h"
#include "sslib/log.h"
#include "sslib/stream_helper.h"
#include "sslib/json.h"
#include "tensor/tensor_util.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using namespace std;
using namespace sslib;

//static
bool ModelWriter::Save(TransformerModel &model, const string &file_path)
{
    (void)model; (void)file_path;
    bool ret = true;
    return ret;
}

//static
bool ModelWriter::Print(const TransformerModel &model, const string &output_dir,
    const string &output_name, bool include_tensor_weights, bool include_tensor_stat)
{
    string path_prefix = output_dir + "/" + output_name;

    string file_path = path_prefix + ".tokenizer.json";
    bool ret = PrintTokenizer(model, file_path);

    file_path = path_prefix + ".json";
    ofstream writer(file_path, ios::binary);
    if (!writer) {
        LogError("Failed to open file %s", file_path.c_str());
        return false;
    }

    writer << "{";
    writer << "\n\"name\":";
    AppendJsonValue(writer, model.spec.sid);

    writer << ",\n\"hyper_params:\":";
    HyperParamsToJson(writer, model.spec.hyper_params);

    writer << ",\n\"vocabulary\":";
    VocabularyToJson(writer, model.vocabulary);

    writer << ",\n\"tensor_spec_table\":";
    TensorSpecTableToJson(writer, model.tensor_spec_table);

    writer << "\n}";
    writer.close();

    const auto &host_net = model.std_network.host_net;

    bool include_data = include_tensor_weights;
    if (model.encoder_embeddings != nullptr)
    {
        file_path = path_prefix + ".embeddings.json";
        LogKeyInfo("Printing encoder token embeddings...");
        PrintTensor(file_path, *model.encoder_embeddings, include_data, include_tensor_stat);
    }

    if (model.decoder_embeddings != nullptr)
    {
        file_path = path_prefix + ".embeddings.json";
        LogKeyInfo("Printing decoder token embeddings...");
        PrintTensor(file_path, *model.decoder_embeddings, include_data, include_tensor_stat);
    }

    if (host_net.encoder_input_norm != nullptr)
    {
        file_path = path_prefix + ".enc.input_norm.json";
        LogKeyInfo("Printing the encoder_input_norm tensor...");
        PrintTensor(file_path, *host_net.encoder_input_norm, include_data, include_tensor_stat);
    }

    if (host_net.decoder_input_norm != nullptr)
    {
        file_path = path_prefix + ".dec.input_norm.json";
        LogKeyInfo("Printing the decoder_input_norm tensor...");
        PrintTensor(file_path, *host_net.decoder_input_norm, include_data, include_tensor_stat);
    }

    if (host_net.output != nullptr)
    {
        file_path = path_prefix + ".output.json";
        LogKeyInfo("Printing the output tensor...");
        PrintTensor(file_path, *host_net.output, include_data, include_tensor_stat);
    }

    int layer_count = (int)host_net.decoder_layers.size();
    int start_layer = 0, end_layer = layer_count;
    for (int layer_idx = start_layer; layer_idx < end_layer; layer_idx++)
    {
        const auto &layer = *host_net.decoder_layers[layer_idx];
        file_path = path_prefix + ".layer_" + to_string(layer_idx) + ".json";
        if (layer_idx % 4 == 0)
        {
            LogKeyInfo("Printing layer %d...", layer_idx);
            PrintLayer(file_path, layer.self_attn, "decoder.self_attn.",
                include_data, include_tensor_stat);
            PrintLayer(file_path, layer.ffn, "decoder.ffn.",
                include_data, include_tensor_stat);
        }
    }

    return ret && writer.good();
}

//static
bool ModelWriter::PrintTokenizer(const TransformerModel &model, const string &file_path)
{
    ofstream writer(file_path, ios::binary);
    if (!writer) {
        LogError("Failed to open file %s", file_path.c_str());
        return false;
    }

    VocabularyToJson(writer, model.vocabulary);

    writer.close();
    return writer.good();
}

//static
void ModelWriter::HyperParamsToJson(ostream &writer,
    const ModelHyperParams &hparams)
{
    writer << "{";
    writer << "\"vocab_size\":" << hparams.vocab_size;
    writer << ", \"embd_dims\":" << hparams.embd_dims;
    writer << ", \"encoder_heads\":" << hparams.encoder_heads;
    writer << ", \"encoder_layers\":" << hparams.encoder_layers;
    writer << ", \"decoder_heads\":" << hparams.decoder_heads;
    writer << ", \"decoder_layers\":" << hparams.decoder_layers;
    writer << ", \"training_context_len\":" << hparams.training_context_len;
    writer << "}";
}

//static
void ModelWriter::VocabularyToJson(ostream &writer,
    const StdVocabulary &vocab)
{
    writer << "{";
    writer << "\n  \"model\":{";
    writer << "\n    \"vocab\":{";

    bool is_first = true;
    for (int token_idx = 0; token_idx < (int)vocab.token_array.size(); token_idx++)
    {
        const auto &token = vocab.token_array[token_idx];

        writer << (is_first ? "\n      \"" : ",\n      \"");
        writer << JsonBuilder::EncodeString(token.str) << "\":";
        writer << token.id;

        is_first = false;
    }

    writer << "\n    }";
    writer << "\n  }";
    writer << "\n}";
}

//static
void ModelWriter::TensorSpecTableToJson(ostream &writer,
    const TensorSpecTable &table)
{
    int tensor_num = (int)table.tensor_array.size();

    writer << "[";
    for (int idx = 0; idx < tensor_num; idx++)
    {
        const auto &tensor_spec = table.tensor_array[idx];
        writer << (idx > 0 ? ",\n  " : "\n  ");
        TensorSpecToJson(writer, tensor_spec);
    }
    writer << "]";
}

//static
void ModelWriter::TensorSpecToJson(ostream &writer,
    const TensorSpec &spec)
{
    writer << "{\"id\":" << spec.id;

    writer << ", \"name\":";
    AppendJsonValue(writer, spec.name);

    writer << ", \"dims\":[";
    for (int dim_idx = 0; dim_idx < spec.dims; dim_idx++)
    {
        writer << (dim_idx > 0 ? ", " : "");
	writer << spec.ne[dim_idx];
    }
    writer << "]";

    writer << ", \"data_type\":" << spec.data_type;
    writer << ", \"size\":" << spec.size;
    writer << ", \"file_idx\":" << spec.file_idx;
    writer << ", \"offset_in_file\":" << spec.offset_in_file;

    writer << "}";
}

//static
bool ModelWriter::PrintTensor(const string &file_path, const HostTensor &tensor,
    bool include_data, bool include_stat)
{
    ofstream strm(file_path);
    Macro_RetFalseIf(!strm);

    TensorUtil::TensorToJson(strm, tensor, include_data, false, include_stat);

    strm.close();
    return strm.good();
}

//static
bool ModelWriter::PrintLayer(const string &file_path,
    const StdHostNetwork::AttentionLayer &layer, const string &title_prefix,
    bool include_data, bool include_stat)
{
    ofstream strm(file_path);
    Macro_RetFalseIf(!strm);

    if (layer.pre_norm != nullptr)
    {
        strm << "\n" << title_prefix << "pre_norm:\n";
        TensorUtil::TensorToJson(strm, *layer.pre_norm,
            include_data, false, include_stat);
    }

    if (layer.qkv != nullptr)
    {
        strm << "\n" << title_prefix << "qkv:\n";
        TensorUtil::TensorToJson(strm, *layer.qkv, include_data, false, include_stat);
    }

    if (layer.wq != nullptr)
    {
        strm << "\n" << title_prefix << "wq:\n";
        TensorUtil::TensorToJson(strm, *layer.wq, include_data, false, include_stat);
    }

    if (layer.wk != nullptr)
    {
        strm << "\n" << title_prefix << "wk:\n";
        TensorUtil::TensorToJson(strm, *layer.wk, include_data, false, include_stat);
    }

    if (layer.wv != nullptr)
    {
        strm << "\n" << title_prefix << "wv:\n";
        TensorUtil::TensorToJson(strm, *layer.wv, include_data, false, include_stat);
    }

    if (layer.wo != nullptr)
    {
        strm << "\n" << title_prefix << "wo:\n";
        TensorUtil::TensorToJson(strm, *layer.wo, include_data, false, include_stat);
    }

    return strm.good();
}

//static
bool ModelWriter::PrintLayer(const string &file_path,
    const StdHostNetwork::FeedForwardLayer &layer, const string &title_prefix,
    bool include_data, bool include_stat)
{
    ofstream strm(file_path);
    Macro_RetFalseIf(!strm);

    if (layer.pre_norm != nullptr)
    {
        strm << "\n" << title_prefix << "pre_norm:\n";
        TensorUtil::TensorToJson(strm, *layer.pre_norm,
            include_data, false, include_stat);
    }

    if (layer.post_norm != nullptr)
    {
        strm << "\n" << title_prefix << "post_norm:\n";
        TensorUtil::TensorToJson(strm, *layer.post_norm, include_data, false, include_stat);
    }

    if (layer.w1 != nullptr)
    {
        strm << "\n" << title_prefix << "w1:\n";
        TensorUtil::TensorToJson(strm, *layer.w1, include_data, false, include_stat);
    }

    if (layer.w2 != nullptr)
    {
        strm << "\n" << title_prefix << "w2:\n";
        TensorUtil::TensorToJson(strm, *layer.w2, include_data, false, include_stat);
    }

    if (layer.w3 != nullptr)
    {
        strm << "\n" << title_prefix << "w3:\n";
        TensorUtil::TensorToJson(strm, *layer.w3, include_data, false, include_stat);
    }

    if (layer.w1n3 != nullptr)
    {
        strm << "\n" << title_prefix << "w1n3:\n";
        TensorUtil::TensorToJson(strm, *layer.w1n3, include_data, false, include_stat);
    }

    return strm.good();
}

//static
void ModelWriter::AppendJsonValue(ostream &writer, const string &utf8_str)
{
    wstring wstr = StringUtil::Utf8ToWideStr(utf8_str);
    writer << "\"" << StringUtil::ToUtf8(JsonBuilder::EncodeString(wstr)) << "\"";
}

TRANSFORMER_END
INFER_FLOW_END
