#pragma once

#include "sslib/vector_ex.h"
#include "tensor/host_float_buffer.h"
#include "transformer_types.h"
#include "inference_types.h"
#if defined(USE_CUDA)
#   include "tensor/device_tensor_builder.h"
#endif

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using sslib::PtrVector;

struct LayerAssignment
{
    const vector<int> *devices = nullptr;
    int start_layer = 0;
    int end_layer = 0;
};

// Model partition among devices
struct ModelPartition
{
    vector<LayerAssignment> encoder_assignments;
    vector<LayerAssignment> decoder_assignments;
};

struct TensorNameInfo
{
    bool is_dec = true;
    int layer_id = 0;
    LayerType layer_type = LayerType::SELF_ATTN;
    LayerTensorId tensor_id = LayerTensorId::ATTN_PRE_NORM;
};

class NetworkBuilder
{
public:
    NetworkBuilder();
    virtual ~NetworkBuilder();
    void Clear();

    bool Init(const InferenceConfig &config, const InferenceConfigEx &config_ex,
        TransformerContext &ctx);

    bool InitNetworkStructure(StdNetwork &net, const ModelSpec &spec) const;

    bool BuildHostNetwork(TransformerModel &model, const ModelSpec &spec, bool is_cpu_only);
    bool BuildHostNetwork_Std(TransformerModel &model, const ModelSpec &spec, bool is_cpu_only);

    bool BuildGgmlNetwork(TransformerModel &model, const ModelSpec &spec, ggml_context *&ctx,
        int encoder_layer_count, int decoder_layer_count);

    static void BuildHostTensorMap(StdHostNetwork &net);

    bool CheckHostModel(const TransformerModel &model, bool is_cpu_only) const;
#if defined(USE_CUDA)
    bool CheckDeviceModel(const TransformerModel &model) const;
    bool CheckDeviceModel(const StdDeviceNetwork &net, const ModelSpec &model_spec,
        int sub_net_idx = -1) const;
#endif //USE_CUDA

#if defined(USE_CUDA)
    bool InitDeviceNetStructure(StdNetwork &net, const ModelSpec &spec) const;

    bool BuildDeviceNetwork(TransformerModel &model, const ModelSpec &spec,
        const ModelPartition &model_partition, int builder_count);

    static bool ParseTensorName(TensorNameInfo &tni, const string &tensor_name,
        const StdDeviceNetwork &device_net, const ModelSpec &spec);

    bool BuildDeviceTensor(StdDeviceNetwork &device_net, const HostTensor &cpu_tensor,
        const TensorNameInfo &tni, const ModelSpec &spec);
    //bool BuildDeviceTensor(StdDeviceNetwork &device_net, const HostTensor &cpu_tensor,
    //    const StdHostNetwork &host_net, const string &tensor_name,
    //    const ModelSpec &spec, const ModelPartition &model_partition);

    DeviceTensor* BuildDeviceTensor_Quant(const HostTensor &cpu_tensor,
        HostHalfBuffer &aux_buffer, const ModelSpec &model_spec,
        bool be_trans = false, DeviceTensor *aux_tensor = nullptr);

    //force dequant
    DeviceTensor* BuildDeviceTensor_ForceDequant(const HostTensor &cpu_tensor,
        HostHalfBuffer &aux_buffer, const ModelSpec &model_spec,
        bool be_trans = false, DeviceTensor *aux_tensor = nullptr);
    bool BuildDeviceTensors_ForceDequant(PtrVector<DeviceTensorEx> &tensor_list,
        const HostTensor &cpu_tensor, HostHalfBuffer &aux_buffer,
        const ModelSpec &model_spec, const vector<int> &device_group,
        bool be_trans = false,
        TensorPartitionType partition_type = TensorPartitionType::BY_COL,
        DeviceTensor *aux_tensor = nullptr);
#endif

    static void SplitGpuLayers(vector<LayerAssignment> &layer_assignments,
        int start_layer, int end_layer, const ModelSpec &spec);
    static void GetDeviceAssignments(vector<LayerAssignment> &layer_assignments,
        const ModelSpec &model_spec, bool is_encoder, int start_gpu_layer = 0);

    static bool ConvertGgmlToHost(HostTensor &host_tensor, const ggml_tensor *ggml_tensor);

    void ClearDeviceMemory();

protected:
    const InferenceConfig *config_ = nullptr;
    const InferenceConfigEx *config_ex_ = nullptr;
    TransformerContext *context_ = nullptr;

    map<LayerTensorId, int> attn_tensor_id_map_;
    map<LayerTensorId, int> ffn_tensor_id_map_;

#if defined(USE_CUDA)
    DeviceTensorBuilder device_tensor_builder_;
    bool is_device_tensor_builder_initialized_ = false;
#endif //USE_CUDA

protected:
    void SetAttentionLayer(StdHostNetwork::AttentionLayer &layer,
        const TransformerModel &model, const char *prefix);
    void SetFeedForwardLayer(StdHostNetwork::FeedForwardLayer &layer,
        const TransformerModel &model, const char *prefix);

    void BuildAttnTensorIdMap(map<LayerTensorId, int> &tensor_id_map);
    void BuildFfnTensorIdMap(map<LayerTensorId, int> &tensor_id_map);

    static void BuildLayerTensorMap(StdHostNetwork::AttentionLayer &layer);
    static void BuildLayerTensorMap(StdHostNetwork::FeedForwardLayer &layer);

    bool CheckModelLayer(const StdHostNetwork::AttentionLayer &layer,
        int layer_id, bool is_encoder, bool is_self_attn) const;
    bool CheckModelLayer(const StdHostNetwork::FeedForwardLayer &layer,
        int layer_id, bool is_encoder) const;

#if defined(USE_CUDA)
    bool CheckModelLayer(const StdDeviceNetwork::AttentionLayer &layer,
        int layer_id, bool is_encoder, bool is_self_attn) const;
    bool CheckModelLayer(const StdDeviceNetwork::FeedForwardLayer &layer,
        int layer_id, bool is_encoder) const;
#endif //USE_CUDA

    ggml_tensor* ConvertHostToGgml(const HostTensor *host_tensor, ggml_context* &ctx);
    bool BuildGgmlNetwork_EncoderLayer(StdGgmlNetwork::EncoderLayer *ggml_layer,
        const StdHostNetwork::EncoderLayer *host_layer, 
        ggml_context* &ctx, const ModelSpec &spec);
    bool BuildGgmlNetwork_DecoderLayer(StdGgmlNetwork::DecoderLayer *ggml_layer,
        const StdHostNetwork::DecoderLayer *host_layer, 
        ggml_context* &ctx, const ModelSpec &spec);
    bool BuildGgmlNetwork_SimpleLayer(StdGgmlNetwork::SimpleLayer &ggml_layer,
        const StdHostNetwork::SimpleLayer &host_layer, 
        ggml_context* &ctx);
    bool BuildGgmlNetwork_AtomicLayer(StdGgmlNetwork::AtomicLayer &ggml_layer,
        const StdHostNetwork::AtomicLayer &host_layer, 
        ggml_context* &ctx);

protected:
#if defined(USE_CUDA)
    bool BuildDeviceNetwork_ByLayer(StdDeviceNetwork &device_net,
        TransformerModel &model, const StdHostNetwork &host_net,
        int builder_count);
    bool BuildDeviceNetwork_ByTensor(PtrVector<StdDeviceNetwork> &device_sub_nets,
        TransformerModel &model, const StdHostNetwork &host_net,
        const LayerAssignment &la, int builder_count);

    bool BuildDeviceNetwork_Embd(StdDeviceNetwork &device_net,
        TransformerModel &model, const StdHostNetwork &host_net,
        HostHalfBuffer &aux_buffer);
    bool BuildDeviceNetwork_SimpleLayer(StdDeviceNetwork::SimpleLayer &device_layer,
        TransformerModel &model, const StdHostNetwork::SimpleLayer &host_layer,
        HostHalfBuffer &aux_buffer);
    bool BuildDeviceNetwork_EncoderOut(StdDeviceNetwork &device_net,
        TransformerModel &model, const StdHostNetwork &host_net,
        HostHalfBuffer &aux_buffer, DeviceTensor &aux_tensor);
    bool BuildDeviceNetwork_DecoderOut(StdDeviceNetwork &device_net,
        TransformerModel &model, const StdHostNetwork &host_net,
        HostHalfBuffer &aux_buffer, DeviceTensor &aux_tensor);
    bool BuildDeviceNetwork_Input(StdDeviceNetwork &device_net,
        TransformerModel &model, const StdHostNetwork &host_net,
        HostHalfBuffer &aux_buffer);

    bool BuildDeviceNets_EncoderOut(PtrVector<StdDeviceNetwork> &device_sub_nets,
        TransformerModel &model, const StdHostNetwork &host_net,
        const vector<int> &device_group, HostHalfBuffer &aux_buffer);
    bool BuildDeviceNets_DecoderOut(PtrVector<StdDeviceNetwork> &device_sub_nets,
        TransformerModel &model, const StdHostNetwork &host_net,
        const vector<int> &device_group, HostHalfBuffer &aux_buffer,
        DeviceTensor &aux_tensor);
    bool BuildDeviceNets_Input(PtrVector<StdDeviceNetwork> &device_sub_nets,
        TransformerModel &model, const StdHostNetwork &host_net,
        const vector<int> &device_group, HostHalfBuffer &aux_buffer);

    bool AddLayerTasks_Std(DeviceTensorBuilder &builder,
        StdDeviceNetwork::AttentionLayer &gpu_layer,
        const StdHostNetwork::AttentionLayer &cpu_layer,
        const ModelSpec &model_spec, int layer_id, bool be_trans,
        bool is_decoder, bool is_cross_attention);
    bool AddLayerTasks_Std(DeviceTensorBuilder &builder,
        StdDeviceNetwork::FeedForwardLayer &gpu_layer,
        const StdHostNetwork::FeedForwardLayer &cpu_layer,
        const ModelSpec &model_spec, int layer_id, bool be_trans);
    bool AddLayerTasks_TensorParallel(DeviceTensorBuilder &builder,
        vector<StdDeviceNetwork::SubLayer*> &gpu_sub_net_layers,
        const StdHostNetwork::SubLayer &cpu_layer,
        const ModelSpec &model_spec, int layer_id, bool be_trans,
        bool is_ffn);

    void BuildTask_Std(DeviceTensorBuilder::Task &task,
        DeviceTensorEx &target_tensor, const HostTensor &cpu_tensor,
        int tensor_type, const ModelSpec &model_spec,
        bool be_trans);

    static void BuildLayerTensorMap(StdDeviceNetwork::AttentionLayer &layer);
    static void BuildLayerTensorMap(StdDeviceNetwork::FeedForwardLayer &layer);
#endif //USE_CUDA
};

TRANSFORMER_END
INFER_FLOW_END
