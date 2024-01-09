#pragma once

#include "sslib/task_monitor.h"
#include "sslib/thread.h"
#include "tensor/device_memory_heap.h"
#include "tensor/cublas_engine.h"
#include "tensor/tensor_opr.h"
#include "kv_cache.h"
#include "inference_types.h"
#include "query_state_table.h"
#include "gpu_inf_global_data.h"

INFER_FLOW_BEGIN
TRANSFORMER_BEGIN

using std::ostream;
using std::mutex;
using std::ostream;
using sslib::PairUInt32;
using sslib::TaskMonitor;
using sslib::Thread;

class GpuInferenceWorker : public sslib::Thread
{
public:
    GpuInferenceWorker() {};
    virtual ~GpuInferenceWorker();

    bool Init(int id, int worker_num, int group_id, int group_num, const InferenceConfig &cfg,
        const InferenceConfigEx &config_ex, TransformerModel &model, int device_id,
        const LayerRange encoder_layer_range, const LayerRange decoder_layer_range,
        bool is_by_layer);

    bool SetGlobalData(GpuInfGlobalData &global_data, const vector<int> &devices);

    void SetInput(const DeviceTensor *input, InferencePerfStat &perf_stat,
        ostream *tensor_writer, const vector<QueryProcInput> &query_list,
        const vector<int> &token_list, int global_end_layer = 0,
        bool is_encoder = false);

    int StartLayer(bool is_encoder) const {
        return is_encoder ? encoder_layer_range_.start : decoder_layer_range_.start;
    }

    int EndLayer(bool is_encoder) const {
        return is_encoder ? encoder_layer_range_.end : decoder_layer_range_.end;
    }

    int device_id() const {
        return device_id_;
    }

    DeviceTensor* GetOutput() {
        return output_tensor_;
    }

    virtual void Run() override;
    virtual void CancelThread() override;

protected:
    struct QueryProcData
    {
        KVCache kv_cache;
        KVCache cross_attn_kv_cache;
    };

    struct CurQKV
    {
        DeviceTensor *q = nullptr;
        DeviceTensor *k = nullptr;
        DeviceTensor *v = nullptr;
    };

    enum class PhaseId
    {
        LAYER_START = 10,
        SELF_ATTN = 11,
        FFN,
        LAYER_END = 29
    };

    struct InputKV
    {
        vector<PairUInt32> query_list;
        DeviceTensor *tensor = nullptr;
    };

    struct AttentionOutput
    {
        DeviceTensor *output = nullptr;
        DeviceTensor *pre_norm = nullptr;
    };

protected:
    int id_ = 0;
    int worker_num_ = 1; //worker count in this group
    int group_id_ = 0;
    int group_num_ = 1;
    const InferenceConfig *config_ = nullptr;
    const InferenceConfigEx *config_ex_ = nullptr;
    TransformerModel *model_ptr_ = nullptr;
    int device_id_ = 0;
    LayerRange encoder_layer_range_;
    LayerRange decoder_layer_range_;
    int global_encoder_end_layer_ = 0;
    int global_decoder_end_layer_ = 0;
    bool is_by_layer_ = true;
    bool is_quant_tensor_exchange_ = true;
    int layer_idx_for_study_ = 0;

    vector<int> devices_;
    GpuInfGlobalData *global_data_ = nullptr;

    PtrVector<QueryProcData> query_proc_data_list_;

    DeviceTensor soft_max_aux_tensor_;
    //StdDeviceNetwork::EncoderLayer encoder_dequant_layer_;
    //StdDeviceNetwork::DecoderLayer std_dequant_layer_;
    DeviceTensor dequant_tensor_;
    DeviceTensorEx dequant_tensor_ex_;

    bool is_cublas_engine_initialized_ = false;
    CublasEngine cublas_engine_;

    //allocators for the current call of evaluate
    const static int LOCAL_DEVICE_HEAP_NUM = 2;
    DeviceMemoryHeap local_device_heaps_[LOCAL_DEVICE_HEAP_NUM];
    DeviceMemoryHeap local_device_heap_;
    DeviceMemoryHeap layer_local_device_heap_;
    BlockedAllocator<DeviceTensor> local_device_tensor_heap_;
    DeviceTensor k_cache_item_, v_cache_item_;

    HostHalfBuffer aux_buffer_;

    //input and output
    bool is_encoder_ = false;
    const DeviceTensor *input_tensor_ = nullptr;
    vector<QueryProcInput> query_list_;
    vector<int> token_id_list_;
    int *device_token_id_array_ = nullptr;
    InferencePerfStat *perf_stat_ = nullptr;
    ostream *tensor_writer_ = nullptr;
    DeviceTensor *output_tensor_ = nullptr;

protected:
    const StdDeviceNetwork* GetDeviceNet() const;
    void BuildInputKV(InputKV &input_kv, ElementType data_type);

    bool GetEmbeddingTensor(DeviceTensor &embd_tensor, bool is_encoder, bool is_pos);

    DeviceTensor* ProcessPreLayer(const DeviceTensor *layer_input, int heap_idx);
    DeviceTensor* ProcessPostLayer(DeviceTensor *layer_input, int heap_idx);

    DeviceTensor* ProcessOutputTransLayer(DeviceTensor *layer_input, int heap_idx);

    DeviceTensor* ProcessGpuLayer(int layer_idx, const DeviceTensor *layer_input,
        const InputKV &input_kv, int heap_idx);

    AttentionOutput ProcessGpuLayer_Attention(int layer_idx,
        const StdDeviceNetwork::AttentionLayer &layer, const DeviceTensor *input_q,
        const InputKV *input_kv, int heap_idx, bool is_encoder);
    DeviceTensor* ProcessGpuLayer_FeedForward(
        int layer_idx, const StdDeviceNetwork::FeedForwardLayer &layer,
        DeviceTensor *input_tensor, int heap_idx, bool is_encoder);

    bool Attention_CalculateCurQKV(CurQKV &cur_qkv, int layer_idx,
        const StdDeviceNetwork::AttentionLayer &layer,
        const DeviceTensor *input_q, const InputKV *input_kv,
        int heap_idx, bool is_encoder);

    DeviceTensor* DistributeAndMergeTensors(const DeviceTensor *tensor,
        bool is_sum, int cur_phase, int next_phase, int heap_idx,
        bool is_study_mode = false);
    void DistributeAndMergeTensors(DeviceTensor *merged_tensor,
        const vector<SourceTensor> &source_tensors,
        int cur_phase, int next_phase);
    DeviceTensor* MergeTensors(const vector<TensorWithDeviceId> &tensor_list,
        bool is_sum, int target_device_id, int heap_idx, bool is_study_mode = false);
    bool DeviceCopy(void *dst, int dst_device, const void *src, int src_device, int bytes);

    void ClearLocalMemory();
    void ClearLayerLocalMemory();

    bool MatrixMultiplication(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &B, bool is_b_column_major,
        const DeviceTensor *bias = nullptr);
    bool MatrixMultiplicationEx(DeviceTensor &C, const DeviceTensor &A,
        const DeviceTensor &AQ, const DeviceTensorEx &B,
        bool is_b_column_major, const DeviceTensor *bias = nullptr);

    //bool BuildDequantLayer();
    bool DequantizeLayer(StdDeviceNetwork::EncoderLayer &target,
        const StdDeviceNetwork::EncoderLayer &source, int layer_id) const;
    bool DequantizeLayer(StdDeviceNetwork::DecoderLayer &target,
        const StdDeviceNetwork::DecoderLayer &source, int layer_id) const;
    bool DequantizeLayer(StdDeviceNetwork::AttentionLayer &target,
        const StdDeviceNetwork::AttentionLayer &source, int layer_id) const;
    bool DequantizeLayer(StdDeviceNetwork::FeedForwardLayer &target,
        const StdDeviceNetwork::FeedForwardLayer &source, int layer_id) const;
    bool DequantizeTensor(DeviceTensorEx &target, const DeviceTensorEx &source,
        bool be_transpose = false, bool be_sync = true) const;

    DeviceTensor* CreateLocalTensor(const DeviceTensor &ref_tensor,
        bool is_layer_local, int heap_idx);
    DeviceTensor* CreateLocalTensor(ElementType etype, int ne0,
        int ne1, int ne2, bool is_layer_local, int heap_idx);
    DeviceTensor* CreateTensor(ElementType etype, const DeviceTensor &ref_tensor,
        bool be_transpose = false);

    DeviceTensor* CreateView(DeviceTensor &ref_tensor, int y_start, int y_count);

    void PrintTensor(const DeviceTensor *tensor, int max_cx, int max_cy,
        int max_cz, const char *title, int layer_id = -1);
    void UpdatePerfStat(int key, const TaskMonitor &tm);
    void UpdatePerfStat(int key, float value);

    bool GetUseGemv(const DeviceTensor &input_tensor) const;
    bool GetUseFullQuantGemv(const DeviceTensor &input_tensor,
        const DeviceTensor &weight_tensor) const;
};

TRANSFORMER_END
INFER_FLOW_END
