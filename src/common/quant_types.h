#pragma once

#include "data_types.h"

INFER_FLOW_BEGIN

// 8-bit quantization with block capacity 32,
// configuration type T1 (16-bit base and 16-bit scale)
// Effectively 9 bits per data item
#define Q8B32_CAPACITY 32
struct BlockQ8_B32T1
{
    uint16_t base;
    uint16_t scale;
    uint8_t data[Q8B32_CAPACITY];
};

// 8-bit quantization with block capacity 32,
// configuration type T2 (16-bit scale, base = 0)
// Effectively 8.5 bits per data item
// (2 + 32) * 8 / 32 = 8.5
#define Q8B32_CAPACITY 32
struct BlockQ8_B32T2
{
    inferflow_fp16 scale;
    int8_t data[Q8B32_CAPACITY];
};

#define Q8B16_CAPACITY 32

// 6-bit quantization with block size 64
// Effectively 6.5 bits per item
// (2 + 2 + 16 + 32) * 8 / 64 = 6.5
#define Q6_B64_CAPACITY 64
struct BlockQ6_B64T1
{
    uint16_t base;  //2 bytes
    uint16_t scale; //2 bytes
    uint8_t data_h[Q6_B64_CAPACITY / 4]; //higher 2 bits of each value
    uint8_t data[Q6_B64_CAPACITY / 2]; //lower 4 bits of each value
};

#define Q5B32_CAPACITY 32
struct BlockQ5_B32T1
{
    uint8_t scale[2];   //scale
    uint8_t base[2];    //minimal value
    uint8_t h_data[Q5B32_CAPACITY / 8]; //the highest bit of each value
    uint8_t data[Q5B32_CAPACITY / 2]; //lower 4 bits of each value
};

// 4-bit quantization with block size 32
// Effectively 5 bits per data item
// (2 + 2 + 16) * 8 / 32 = 5
#define Q4B32_CAPACITY 32
//#pragma pack(push, 1) //!!! Do NOT set alignment
struct BlockQ4_B32T1
{
    uint16_t base;
    uint16_t scale;
    uint8_t data[Q4B32_CAPACITY / 2];
};
//#pragma pack(pop) //!!! Do NOT set alignment

// 4-bit quantization with block size 32
// Effectively 4.5 bits per data item
// (1 + 1 + 16) * 8 / 32 = 4.5
struct BlockQ4_B32T2
{
    uint8_t base;
    uint8_t scale;
    uint8_t data[Q4B32_CAPACITY / 2]; //16
};

// 4-bit quantization with block size 16
// Effectively 5 bits per item
// (1 + 1 + 8) * 8 / 16 = 5
#define Q4B16_CAPACITY 16
struct BlockQ4_B16
{
    uint8_t base;
    uint8_t scale;
    uint8_t data[Q4B16_CAPACITY / 2];
};

// 3.5-bit quantization with block size 64
// Effectively 4 bits per data item
// (2 + 2 + 4 + 8 + 16) * 8 / 64 = 4
#define Q3H_B64_CAPACITY 64
struct BlockQ3H_B64T1
{
    uint16_t base;  //2 bytes
    uint16_t scale; //2 bytes
    uint8_t data_h[Q3H_B64_CAPACITY / 16];  //4 bytes
    uint8_t data_m[Q3H_B64_CAPACITY / 8];   //8 bytes
    uint8_t data[Q3H_B64_CAPACITY / 4];     //16 bytes
};

// 3-bit quantization with block size 32
// Effectively 4 bits per data item
// (2 + 2 + 4 + 8) * 8 / 32 = 4
#define Q3B32_CAPACITY 32
struct BlockQ3_B32T1
{
    uint16_t base;  //2 bytes
    uint16_t scale; //2 bytes
    uint8_t h_data[Q3B32_CAPACITY / 8]; //4 bytes
    uint8_t data[Q3B32_CAPACITY / 4];   //8 bytes
};

// 3-bit quantization with block size 32
// Effectively 3.5 bits per data item
// (2 + 4 + 8) * 8 / 32 = 3.5
struct BlockQ3_B32T2
{
    uint16_t params; //10-bit scale and 6-bit z value
    uint8_t h_data[Q3B32_CAPACITY / 8]; //4 bytes
    uint8_t data[Q3B32_CAPACITY / 4];   //8 bytes
};

// 3-bit quantization with block size 16
// Effectively 3.5 bits per data item
// (1 + 1 + 4 + 8) * 8 / 32 = 3.5
#define Q3B16_CAPACITY 16
struct BlockQ3_B16T1
{
    uint8_t base;  //1 byte
    uint8_t scale; //1 byte
    uint8_t h_data[Q3B32_CAPACITY / 8]; //4 bytes
    uint8_t data[Q3B32_CAPACITY / 4];   //8 bytes
};

// 2-bit quantization with block size 32
// Effectively 3 bits per data item
// (2 + 2 + 8) * 8 / 32 = 3
#define Q2B32_CAPACITY 32
//#pragma pack(push, 1) //!!! Do NOT set alignment
struct BlockQ2_B32T1
{
    uint16_t base;
    uint16_t scale;
    uint8_t data[Q2B32_CAPACITY / 4]; //8
};
//#pragma pack(pop) //!!! Do NOT set alignment

#define Q2B16_CAPACITY 16
struct BlockQ2_B16T1
{
    uint8_t base;
    uint8_t scale;
    uint8_t data[Q2B16_CAPACITY / 4];
};

struct LinearQuantParams
{
    float z = 0; //zero-point
    float scale1 = 0; //scaling factor 1
    float scale2 = 0; //scaling factor 2
};

struct LogQuantParams
{
    float base = 1.1f;
    int scale = 1000;
    int start = 10;
};

INFER_FLOW_END
