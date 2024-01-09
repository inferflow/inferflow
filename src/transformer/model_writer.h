#pragma once

#include "sslib/binary_stream.h"
#include "model.h"
#include "namespace.inc"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::string;
using std::ostream;
using sslib::IBinaryStream;

class ModelWriter
{
public:
    static bool Save(TransformerModel &model, const string &file_path);

    static bool Print(const TransformerModel &model, const string &output_dir,
        const string &output_name, bool include_tensor_weights = false,
        bool include_tensor_stat = false);

    static bool PrintTokenizer(const TransformerModel &model, const string &file_path);

    static void HyperParamsToJson(ostream &writer, const ModelHyperParams &hparams);
    static void VocabularyToJson(ostream &writer, const StdVocabulary &vocab);
    static void TensorSpecTableToJson(ostream &writer, const TensorSpecTable &table);
    static void TensorSpecToJson(ostream &writer, const TensorSpec &spec);

    static bool PrintTensor(const string &file_path, const HostTensor &tensor,
        bool include_data, bool include_stat);
    static bool PrintLayer(const string &file_path, const StdHostNetwork::AttentionLayer &layer,
        const string &title_prefix, bool include_data, bool include_stat);
    static bool PrintLayer(const string &file_path, const StdHostNetwork::FeedForwardLayer &layer,
        const string &title_prefix, bool include_data, bool include_stat);

protected:
    static void AppendJsonValue(ostream &writer, const string &value);
};

TRANSFORMER_END
INFER_FLOW_END
