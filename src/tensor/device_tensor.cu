#include "device_tensor.h"
#include "cuda_runtime.h"
#include "sslib/log.h"
#include "common/quant_cuda.h"
#include <algorithm>

INFER_FLOW_BEGIN

using namespace std;
using namespace sslib;

void RawDeviceArray::Clear()
{
    if (this->data != nullptr)
    {
        if (is_auto_free_) {
            cudaFree(this->data);
        }
        this->data = nullptr;
    }

    this->size = 0;
}

uint64_t RawDeviceArray::ByteCount() const
{
    return TensorCommon::ByteCount(this->data_type, this->size);
}

float RawDeviceArray::MemoryCost_MB() const
{
    return ByteCount() / 1024.0f / 1024;
}

float RawDeviceArray::MemoryCost_GB() const
{
    return ByteCount() / 1024.0f / 1024 / 1024;
}

DeviceTensor::DeviceTensor(bool is_auto_free) : RawDeviceArray(is_auto_free)
{
}

DeviceTensor::~DeviceTensor()
{
    Clear();
}

void DeviceTensor::Clear()
{
    id = 0;
    dim = 0;
    ne[0] = 0;
    ne[1] = 0;
    ne[2] = 0;

    RawDeviceArray::Clear();
}

const float* RawDeviceArray::data_f32() const
{
    return data_type == ElementType::F32 ? (const float*)this->data : nullptr;
}

float* RawDeviceArray::data_f32()
{
    return data_type == ElementType::F32 ? (float*)this->data : nullptr;
}

const half* RawDeviceArray::data_f16() const
{
    return data_type == ElementType::F16 ? (const half*)this->data : nullptr;
}

half* RawDeviceArray::data_f16()
{
    return data_type == ElementType::F16 ? (half*)this->data : nullptr;
}

void RawDeviceArray::Set(float *src_data, int src_size, bool arg_is_auto_free)
{
    this->data_type = ElementType::F32;
    this->data = src_data;
    this->size = src_size;
    this->is_auto_free_ = arg_is_auto_free;
}

void RawDeviceArray::Set(half *src_data, int src_size, bool arg_is_auto_free)
{
    this->data_type = ElementType::F16;
    this->data = src_data;
    this->size = src_size;
    this->is_auto_free_ = arg_is_auto_free;
}

void RawDeviceArray::Set(ElementType etype, void *src_data, int src_size, bool arg_is_auto_free)
{
    this->data_type = etype;
    this->data = src_data;
    this->size = src_size;
    this->is_auto_free_ = arg_is_auto_free;
}

bool RawDeviceArray::New(ElementType etype, int element_count)
{
    Clear();
    this->data_type = etype;

    int element_size = TensorCommon::ElementSize(etype);
    auto ret_code = cudaMalloc((void**)&this->data, element_count * element_size);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to call cudaMalloc: %d (%s)", ret_code, cudaGetErrorString(ret_code));
        data = nullptr;
        return false;
    }

    this->size = element_count;
    return true;
}

bool RawDeviceArray::NewF32(int element_count)
{
    return New(ElementType::F32, element_count);
}

bool RawDeviceArray::NewF16(int element_count)
{
    return New(ElementType::F16, element_count);
}

bool RawDeviceArray::AssignZero()
{
    int element_size = TensorCommon::ElementSize(data_type);
    auto ret_code = cudaMemset(this->data, 0, this->size * element_size);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to call cudaMemset: %d (%s)",
            ret_code, cudaGetErrorString(ret_code));
        return false;
    }

    return true;
}

bool RawDeviceArray::FromHost(const vector<float> &host_vec)
{
    bool ret = FromHost(host_vec.data(), (int)host_vec.size());
    return ret;
}

bool RawDeviceArray::FromHost(const float *host_vec, int element_count)
{
    bool ret = New(ElementType::F32, element_count);
    if (!ret) {
        return false;
    }

    int byte_num = element_count * sizeof(float);
    auto ret_code = cudaMemcpy(this->data, host_vec, byte_num, cudaMemcpyHostToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy data from host to device: %d (%s)",
            ret_code, cudaGetErrorString(ret_code));
        Clear();
        return false;
    }

    this->size = element_count;
    return true;
}

bool RawDeviceArray::FromHost(const vector<half> &host_vec)
{
    bool ret = FromHost(host_vec.data(), (int)host_vec.size());
    return ret;
}

bool RawDeviceArray::FromHost(const half *host_vec, int element_count)
{
    bool ret = New(ElementType::F16, element_count);
    if (!ret) {
        return false;
    }

    int byte_num = element_count * sizeof(half);
    auto ret_code = cudaMemcpy(this->data, host_vec, byte_num, cudaMemcpyHostToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy data from host to device: %d (%s)",
            ret_code, cudaGetErrorString(ret_code));
        Clear();
        return false;
    }

    this->size = element_count;
    return true;
}

bool RawDeviceArray::CopyFromDevice(const float *device_vec, int element_count)
{
    bool ret = New(ElementType::F32, element_count);
    if (!ret) {
        return false;
    }

    int byte_num = element_count * sizeof(float);
    auto ret_code = cudaMemcpy(this->data, device_vec, byte_num, cudaMemcpyDeviceToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy data from device to device: %d (%s)",
            ret_code, cudaGetErrorString(ret_code));
        Clear();
        return false;
    }

    this->size = element_count;
    return true;
};

bool RawDeviceArray::CopyFromDevice(const half *device_vec, int element_count)
{
    bool ret = New(ElementType::F16, element_count);
    if (!ret) {
        return false;
    }

    int byte_num = element_count * sizeof(half);
    auto ret_code = cudaMemcpy(this->data, device_vec, byte_num, cudaMemcpyDeviceToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy data from device to device: %d (%s)",
            ret_code, cudaGetErrorString(ret_code));
        Clear();
        return false;
    }

    this->size = element_count;
    return true;
};

bool DeviceTensor::CopyFromHost(const void *host_data, int bytes)
{
    int rows = Rows();
    uint64_t my_bytes = this->bytes_per_row * rows;
    if (my_bytes != bytes)
    {
        LogError("Inconsistent numbers of bytes: %d vs. %d", my_bytes, bytes);
        return false;
    }

    auto ret_code = cudaMemcpy(this->data, host_data, bytes, cudaMemcpyHostToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy %d byte(s) from host: %d (%s)",
            bytes, ret_code, cudaGetErrorString(ret_code));
        return false;
    }

    return true;
}

bool DeviceTensor::CopyRowFromHost(const void *host_data, int y, int z)
{
    void *target_row = RowData(y, z);
    auto ret_code = cudaMemcpy(target_row, host_data, this->bytes_per_row,
        cudaMemcpyHostToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy row (%d, %d) from host to device: %d (%s)",
            y, z, ret_code, cudaGetErrorString(ret_code));
        return false;
    }

    return true;
}

bool DeviceTensor::CopyFromDevice(const void *source, int bytes)
{
    int rows = Rows();
    uint64_t my_bytes = this->bytes_per_row * rows;
    if (my_bytes != bytes)
    {
        LogError("Inconsistent numbers of bytes: %d vs. %d", my_bytes, bytes);
        return false;
    }

    auto ret_code = cudaMemcpy(this->data, source, bytes, cudaMemcpyDeviceToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy %d byte(s) from device: %d (%s)",
            bytes, ret_code, cudaGetErrorString(ret_code));
        return false;
    }

    return true;
}

bool DeviceTensor::CopyToHost(std::vector<float> &host_vec) const
{
    bool ret = true;
    bool is_fp32 = data_type == ElementType::F32;
    host_vec.resize(this->size, 0);
    if (!this->is_zy_data && is_fp32)
    {
        const void *src_data = this->data;
        int byte_num = sizeof(float) * (int)this->size;
        auto ret_code = cudaMemcpy(host_vec.data(), src_data, byte_num, cudaMemcpyDeviceToHost);
        if (ret_code != cudaSuccess)
        {
            LogError("Failed to copy to host: %d (%s)", ret_code, cudaGetErrorString(ret_code));
            ret = false;
        }
    }
    else
    {
        vector<float> med_vec(this->ne[0], 0);
        for (int z_idx = 0; z_idx < this->ne[2]; z_idx++)
        {
            for (int y_idx = 0; y_idx < this->ne[1]; y_idx++)
            {
                ret = CopyRowToHost(med_vec, y_idx, z_idx);

                int row = z_idx * this->ne[1] + y_idx;
                for (int idx = 0; ret && idx < this->ne[0]; idx++) {
                    host_vec[row * this->ne[0] + idx] = med_vec[idx];
                }
            }
        }
    }

    return ret;
}

bool DeviceTensor::CopyToHost(std::vector<half> &host_vec) const
{
    bool ret = true;
    bool is_fp16 = data_type == ElementType::F16;
    host_vec.resize(this->size, 0);
    if (!this->is_zy_data && is_fp16)
    {
        const void *src_data = this->data;
        int byte_num = sizeof(half) * (int)this->size;
        auto ret_code = cudaMemcpy(host_vec.data(), src_data, byte_num, cudaMemcpyDeviceToHost);
        if (ret_code != cudaSuccess)
        {
            LogError("Failed to copy to host: %d (%s)", ret_code, cudaGetErrorString(ret_code));
            ret = false;
        }
    }
    else
    {
        vector<half> med_vec(this->ne[0], 0);
        for (int z_idx = 0; z_idx < this->ne[2]; z_idx++)
        {
            for (int y_idx = 0; y_idx < this->ne[1]; y_idx++)
            {
                ret = CopyRowToHost(med_vec, y_idx, z_idx);

                int row = z_idx * this->ne[1] + y_idx;
                for (int idx = 0; ret && idx < this->ne[0]; idx++) {
                    host_vec[row * this->ne[0] + idx] = med_vec[idx];
                }
            }
        }
    }

    return ret;
}

bool DeviceTensor::CopyToHost(HostTensor &host_tensor, int start_row, int rows) const
{
    bool ret = true;
    host_tensor.Clear();

    if (start_row == 0 && rows < 0)
    {
        ret = host_tensor.New(this->data_type, this->dim, this->ne);
        Macro_RetxFalseIf(!ret, LogError("Failed to create the host tensor"));

        uint64_t bytes = TensorCommon::ByteCount(this->data_type, this->size);
        ret = CudaUtil::DeviceToHostMemcpy(host_tensor.data, this->data, bytes);
    }
    else
    {
        int src_rows = Rows();
        start_row = min(max(start_row, 0), rows);
        int target_rows = rows >= 0 ? (rows - start_row) : (src_rows - start_row);
        host_tensor.New(this->data_type, this->ne[0], target_rows);

        const void *src_data = this->RowData(start_row);
        uint8_t *target_data = (uint8_t*)host_tensor.data;
        uint64_t bytes = TensorCommon::ByteCount(this->data_type, host_tensor.size);
        ret = CudaUtil::DeviceToHostMemcpy(target_data, src_data, bytes);
    }

    return ret;
}

bool DeviceTensor::CopyRowToHost(std::vector<float> &host_vec, int y, int z) const
{
    bool ret = true;
    uint64_t byte_num = this->bytes_per_row;
    bool is_fp32 = data_type == ElementType::F32;
    const void *row_data = RowData(y, z);
    uint64_t block_num = 0;

    host_vec.resize(this->ne[0], 0);

    cudaError_t ret_code = cudaSuccess;
    if (is_fp32)
    {
        ret_code = cudaMemcpy(host_vec.data(), row_data, byte_num, cudaMemcpyDeviceToHost);
    }
    else
    {
        vector<uint8_t> med_data(byte_num);
        ret_code = cudaMemcpy(med_data.data(), row_data, byte_num, cudaMemcpyDeviceToHost);

        if (ret_code == cudaSuccess)
        {
            const BlockQ8_B32T1 *q8_b32t1_blocks = nullptr;
            const BlockQ8_B32T2 *q8_b32t2_blocks = nullptr;
            const BlockQ6_B64T1 *q6_b64t1_blocks = nullptr;
            const BlockQ5_B32T1 *q5_blocks = nullptr;
            const BlockQ4_B16 *q4b16_blocks = nullptr;
            const BlockQ4_B32T1 *q4b32t1_blocks = nullptr;
            const BlockQ3H_B64T1 *q3h_b64t1_blocks = nullptr;
            const BlockQ3_B32T1 *q3b32t1_blocks = nullptr;
            const BlockQ2_B32T1 *q2b32t1_blocks = nullptr;

            switch (this->data_type)
            {
            case ElementType::F16:
                for (int idx = 0; idx < this->ne[0]; idx++)
                {
                    host_vec[idx] = __half2float(*(const half*)(med_data.data() + idx * sizeof(half)));
                }
                break;
            case ElementType::Q8_B32T1:
                q8_b32t1_blocks = (const BlockQ8_B32T1*)med_data.data();
                block_num = this->bytes_per_row / sizeof(BlockQ8_B32T1);
                Quantization::DequantizeRow_Q8_B32T1(host_vec.data(), q8_b32t1_blocks, block_num);
                break;
            case ElementType::Q8_B32T2:
                q8_b32t2_blocks = (const BlockQ8_B32T2*)med_data.data();
                block_num = this->bytes_per_row / sizeof(BlockQ8_B32T2);
                Quantization::DequantizeRow_Q8_B32T2(host_vec.data(), q8_b32t2_blocks, block_num);
                break;
            case ElementType::Q6_B64T1:
                q6_b64t1_blocks = (const BlockQ6_B64T1*)med_data.data();
                block_num = this->bytes_per_row / sizeof(BlockQ6_B64T1);
                Quantization::DequantizeRow_Q6_B64T1(host_vec.data(), q6_b64t1_blocks, block_num);
                break;
            case ElementType::Q5:
                q5_blocks = (const BlockQ5_B32T1*)med_data.data();
                block_num = this->bytes_per_row / sizeof(BlockQ5_B32T1);
                Quantization::DequantizeRow_Q5(host_vec.data(), q5_blocks, block_num);
                //if (y == 0 && z == 0)
                //{
                //    LogKeyInfo("block_num: %d; delta: %f, base: %f", block_num,
                //        (float)blocks[0].delta, (float)blocks[0].base);
                //}
                break;
            case ElementType::Q4_B16:
                q4b16_blocks = (const BlockQ4_B16*)med_data.data();
                block_num = this->bytes_per_row / sizeof(BlockQ4_B16);
                Quantization::DequantizeRow_Q4_B16(host_vec.data(), q4b16_blocks, block_num);
                break;
            case ElementType::Q4_B32T1A:
            case ElementType::Q4_B32T1B:
                q4b32t1_blocks = (const BlockQ4_B32T1*)med_data.data();
                block_num = this->bytes_per_row / sizeof(BlockQ4_B32T1);
                Quantization::DequantizeRow_Q4_B32T1(host_vec.data(), q4b32t1_blocks, block_num);
                break;
            case ElementType::Q3H_B64T1:
                q3h_b64t1_blocks = (const BlockQ3H_B64T1*)med_data.data();
                block_num = this->bytes_per_row / sizeof(BlockQ3H_B64T1);
                Quantization::DequantizeRow_Q3H_B64T1(host_vec.data(), q3h_b64t1_blocks, block_num);
                break;
            case ElementType::Q3_B32T1A:
            case ElementType::Q3_B32T1B:
                q3b32t1_blocks = (const BlockQ3_B32T1*)med_data.data();
                block_num = this->bytes_per_row / sizeof(BlockQ3_B32T1);
                Quantization::DequantizeRow_Q3_B32T1(host_vec.data(), q3b32t1_blocks, block_num);
                break;
            case ElementType::Q2_B32T1A:
            case ElementType::Q2_B32T1B:
                q2b32t1_blocks = (const BlockQ2_B32T1*)med_data.data();
                block_num = this->bytes_per_row / sizeof(BlockQ2_B32T1);
                Quantization::DequantizeRow_Q2_B32T1(host_vec.data(), q2b32t1_blocks, block_num);
                break;
            default:
                LogError("Element type %d has not been handled yet.", this->data_type);
                ret = false;
                break;
            }
        }
    }

    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy row (%d, %d) to host: %d (%s)",
            y, z, ret_code, cudaGetErrorString(ret_code));
        return false;
    };
    return ret;
}

bool DeviceTensor::CopyRowToHost(std::vector<inferflow_fp16> &host_vec, int y, int z) const
{
    bool ret = true;
    int element_size = TensorCommon::ElementSize(this->data_type);
    int byte_num = element_size * this->ne[0];
    bool is_fp16 = data_type == ElementType::F16;
    const void *row_data = RowData(y, z);

    host_vec.resize(this->ne[0], 0);

    cudaError_t ret_code = cudaSuccess;
    if (is_fp16)
    {
        ret_code = cudaMemcpy(host_vec.data(), row_data, byte_num, cudaMemcpyDeviceToHost);
    }
    else
    {
        vector<uint8_t> med_data(byte_num);
        ret_code = cudaMemcpy(med_data.data(), row_data, byte_num, cudaMemcpyDeviceToHost);
        if (ret_code == cudaSuccess)
        {
            switch (this->data_type)
            {
                case ElementType::F32:
                    for (int idx = 0; idx < this->ne[0]; idx++)
                    {
                        host_vec[idx] = (inferflow_fp16)(*(const float*)(med_data.data() + idx * element_size));
                    }
                    break;
                default:
                    LogError("Element type %d has not been handled yet.", this->data_type);
                    ret = false;
                    break;
            }
        }
    }

    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy row (%d, %d) to host: %d (%s)",
            y, z, ret_code, cudaGetErrorString(ret_code));
        return false;
    };
    return ret;
}

bool DeviceTensor::HasCompatibleStructureWith(const DeviceTensor &rhs, bool be_transpose) const
{
    if (be_transpose)
    {
        return this->ne[0] == rhs.ne[1] && this->ne[1] == rhs.ne[0]
            && this->ne[2] == rhs.ne[2] && this->ne[2] == 1;
    }
    else
    {
        return this->ne[0] == rhs.ne[0] && this->ne[1] == rhs.ne[1]
            && this->ne[2] == rhs.ne[2];
    }
}

bool DeviceTensor::New(ElementType etype, int n0, int n1, int n2)
{
    Clear();

    this->data_type = etype;
    bool ret = SetStructure(n0, n1, n2);
    if (!ret) {
        return false;
    }

    uint64_t bytes = this->bytes_per_row * this->ne[1] * this->ne[2];
    auto ret_code = cudaMalloc((void**)&this->data, bytes);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to call cudaMalloc (%d bytes). Error: %d (%s)",
            bytes, ret_code, cudaGetErrorString(ret_code));
        this->data = nullptr;
        return false;
    }

    return ret;
}

bool DeviceTensor::New(ElementType etype, int input_dim, const int *ne_per_dim)
{
    Clear();

    dim = min(input_dim, MaxDimCount);
    int n0 = dim > 0 && ne_per_dim != nullptr ? ne_per_dim[0] : 0;
    int n1 = dim > 1 && ne_per_dim != nullptr ? ne_per_dim[1] : 0;
    int n2 = dim > 2 && ne_per_dim != nullptr ? ne_per_dim[2] : 0;
    bool ret = New(etype, n0, n1, n2);
    return ret;
}

bool DeviceTensor::FromHost(const float *host_vec, int n0, int n1, int n2)
{
    Clear();

    data_type = ElementType::F32;
    bool ret = SetStructure(n0, n1, n2);
    if (!ret) {
        return false;
    }

    ret = RawDeviceArray::FromHost(host_vec, this->size);
    return ret;
}

bool DeviceTensor::FromHost(const half *host_vec, int n0, int n1, int n2)
{
    Clear();

    data_type = ElementType::F16;
    bool ret = SetStructure(n0, n1, n2);
    if (!ret) {
        return false;
    }

    ret = RawDeviceArray::FromHost(host_vec, this->size);
    return ret;
}

bool DeviceTensor::SetStructure(int n0, int n1, int n2)
{
    if (n0 <= 0 || n1 < 0 || n2 < 0) {
        return false;
    }

    dim = n1 == 0 ? 1 : (n2 == 0 ? 2 : 3);
    this->ne[0] = n0;
    this->ne[1] = max(1, n1);
    this->ne[2] = max(1, n2);
    this->bytes_per_row = TensorCommon::ByteCount(this->data_type, n0);

    this->size = ne[0] * ne[1] * ne[2];
    return true;
}

bool DeviceTensor::SetStructure(int dim_count, const int dims[MaxDimCount])
{
    if (dim_count <= 0 || dim_count > MaxDimCount) {
        return false;
    }

    int new_size = 1;
    for (int idx = 0; idx < MaxDimCount; idx++)
    {
        this->ne[idx] = max(1, dims[idx]);
	new_size *= this->ne[idx];
    }

    this->bytes_per_row = TensorCommon::ByteCount(this->data_type, this->ne[0]);
    this->dim = dim_count;
    this->size = new_size;
    return true;
}

const void* DeviceTensor::RowData(int row) const
{
    //int element_size = TensorCommon::ElementSize(data_type);
    //return (const void*)((const uint8_t*)this->data + row * ne[0] * element_size);
    return (const void*)((const uint8_t*)this->data + row * this->bytes_per_row);
}

void* DeviceTensor::RowData(int row)
{
    //int element_size = TensorCommon::ElementSize(data_type);
    //return (void*)((uint8_t*)this->data + row * ne[0] * element_size);
    return (void*)((uint8_t*)this->data + row * this->bytes_per_row);
}

const void* DeviceTensor::RowData(int y, int z) const
{
    int row = is_zy_data ? (y * ne[2] + z) : (z * ne[1] + y);
    return (const void*)((const uint8_t*)this->data + row * this->bytes_per_row);
}

void* DeviceTensor::RowData(int y, int z)
{
    //int element_size = TensorCommon::ElementSize(data_type);
    //int row = is_zy_data ? (y * ne[2] + z) : (z * ne[1] + y);
    //return (void*)((uint8_t*)this->data + row * ne[0] * element_size);
    int row = is_zy_data ? (y * ne[2] + z) : (z * ne[1] + y);
    return (void*)((uint8_t*)this->data + row * this->bytes_per_row);
}

std::ostream& DeviceTensor::Print(std::ostream &strm, int max_ne0,
    int max_ne1, int max_ne2, const char *title) const
{
    std::vector<float> host_vec_f32;
    std::vector<half> host_vec_f16;
    bool is_f16 = this->data_type == ElementType::F16;
    if (is_f16) {
        this->CopyToHost(host_vec_f16);
    }
    else {
        this->CopyToHost(host_vec_f32);
    }

    if (max_ne0 <= 0) {
        max_ne0 = INT32_MAX;
    }
    if (max_ne1 <= 0) {
        max_ne1 = INT32_MAX;
    }
    if (max_ne2 <= 0) {
        max_ne2 = INT32_MAX;
    }

    if (title != nullptr && strlen(title) != 0)
    {
        strm << title;
    }

    strm << "(" << this->ne[0] << ", " << this->ne[1] << ", " << this->ne[2] << ")\n";

    int n1 = this->ne[0], n2 = this->ne[0] * this->ne[1];
    for (int i2 = 0; i2 < this->ne[2]; i2++)
    {
        strm << (i2 > 0 ? ",\n " : (this->dim >= 3 ? "[" : ""));
        if (i2 >= max_ne2) {
            strm << "...";
            break;
        }

        for (int i1 = 0; i1 < this->ne[1]; i1++)
        {
            strm << (i1 > 0 ? (this->dim >= 3 ? ",\n  " : ",\n ") : (this->dim >= 2 ? "[" : ""));
            if (i1 >= max_ne1) {
                strm << "...";
                break;
            }

            strm << "[";
            for (int i0 = 0; i0 < this->ne[0]; i0++)
            {
                if (i0 >= max_ne0) {
                    strm << "...";
                    break;
		}

                if (i0 > 0) {
                    strm << ", ";
                }

                int idx = i2 * n2 + i1 * n1 + i0;
                if (is_f16) {
                    strm << __half2float(host_vec_f16[idx]);
                }
                else {
                    strm << host_vec_f32[idx];
                }
            } //i0
            strm << "]";
        } //i1

        if (this->dim >= 2) {
            strm << "]";
        }
    } //i2

    if (this->dim >= 3) {
        strm << "]";
    }

    /*int n1 = this->ne[0], n2 = this->ne[0] * this->ne[1];
    int vec_size = is_f16 ? (int)host_vec_f16.size() : host_vec_f32.size();
    for (int idx = 0; idx < vec_size; idx++)
    {
        int i0 = idx % this->ne[0];
        int i1 = idx / this->ne[0] % this->ne[1];
        int i2 = idx / this->ne[0] / this->ne[1] % this->ne[2];
        bool in_box = i0 < max_ne0 && i1 < max_ne1 && i2 < max_ne2;

        if (this->ne[2] > 1 && idx % (n1 * n2) == 0)
        {
            if (in_box) {
                strm << (idx > 0 ? "]],\n [[" : "[[[");
            }
        }
        else if (this->ne[1] > 1 && idx % n1 == 0)
        {
            if (in_box) {
                strm << (idx > 0 ? "],\n [" : "[[");
            }
        }
        else
        {
            if (in_box) {
                strm << (idx > 0 ? ", " : "[");
            }
        }

        if (in_box)
        {
            if (is_f16) {
                strm << __half2float(host_vec_f16[idx]);
            }
            else {
                strm << host_vec_f32[idx];
            }
        }
    }
    strm << (this->ne[2] > 1 ? "]]]" : (this->ne[1] > 1 ? "]]" : "]"));*/

    return strm;
}

std::ostream& operator << (std::ostream &strm, const DeviceTensor &tensor)
{
    tensor.Print(strm);
    return strm;
}

////////////////////////////////////////////////////////////////////////////////
// class DeviceSparseMatrix

DeviceSparseMatrix::DeviceSparseMatrix()
{
}

DeviceSparseMatrix::~DeviceSparseMatrix()
{
    Clear();
}

void DeviceSparseMatrix::Clear()
{
    if (this->data_ != nullptr)
    {
        if (is_auto_free_) {
            cudaFree(this->data_);
        }
        this->data_ = nullptr;
    }

    if (this->row_offset_array_ != nullptr)
    {
        if (is_auto_free_) {
            cudaFree(row_offset_array_);
        }
        this->row_offset_array_ = nullptr;
    }

    this->id_ = 0;
    this->rows_ = 0;
    this->cols_ = 0;
    this->size_ = 0;
}

bool DeviceSparseMatrix::New(int non_zero_cell_count, int cols, int rows)
{
    Clear();

    int bytes = non_zero_cell_count * sizeof(SparseMatrixCell);
    auto ret_code = cudaMalloc((void**)&this->data_, bytes);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to call cudaMalloc: %d (%s)", ret_code, cudaGetErrorString(ret_code));
        data_ = nullptr;
        return false;
    }

    this->size_ = non_zero_cell_count;
    rows_ = rows;
    cols_ = cols;
    return true;
}

bool DeviceSparseMatrix::Set(const vector<SparseMatrixCell> &cells,
    const vector<int> &row_offset_array, int cols, int rows)
{
    int cell_count = (int)cells.size();
    bool ret = New(cell_count, cols, rows);
    Macro_RetFalseIf(!ret);

    int bytes = cell_count * (int)sizeof(SparseMatrixCell);
    auto ret_code = cudaMemcpy(this->data_, cells.data(), bytes, cudaMemcpyHostToDevice);
    if (ret_code != cudaSuccess)
    {
        LogError("Failed to copy data from host to device: %d (%s)",
            ret_code, cudaGetErrorString(ret_code));
        Clear();
        return false;
    }

    bytes = (int)row_offset_array.size() * sizeof(int);
    ret_code = cudaMalloc((void**)&this->row_offset_array_, bytes);
    ret = CudaUtil::CheckReturnCode(ret_code);
    Macro_RetFalseIf(!ret);

    ret_code = cudaMemcpy(this->row_offset_array_, row_offset_array.data(),
        bytes, cudaMemcpyHostToDevice);
    ret = CudaUtil::CheckReturnCode(ret_code);
    Macro_RetFalseIf(!ret);

    return ret;
}
 
bool DeviceSparseMatrix::SetSortedCells(const vector<SparseMatrixCell> &cells, int cols, int rows)
{
    vector<int> row_offset_array(rows, 0);
    int prev_row = -1;
    for (int cell_idx = 0; cell_idx < (int)cells.size(); cell_idx++)
    {
        int row_idx = cells[cell_idx].row;
        if (row_idx > prev_row)
        {
            for (int mid_row = prev_row + 1; mid_row <= row_idx; mid_row++) {
                row_offset_array[mid_row] = cell_idx;
            }

            prev_row = row_idx;
        }
    }

    for (int mid_row = prev_row + 1; mid_row < rows; mid_row++) {
        row_offset_array[mid_row] = (int)cells.size();
    }

    bool ret = this->Set(cells, row_offset_array, cols, rows);
    return ret;
}

INFER_FLOW_END
