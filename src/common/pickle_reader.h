#pragma once

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include "sslib/string_blocked_heap.h"
#include "sslib/binary_stream.h"
#include "namespace.inc"

INFER_FLOW_BEGIN

using std::string;
using std::vector;
using std::map;
using std::ostream;
using sslib::StringBlockedHeap;
using sslib::IBinaryStream;

//
// Put, BinPut, LongBinPut: Store the stack top into the memo. The stack is not popped.
//    Put: Arg1 (the index of the memo location) is a newline-terminated decimal string
//    BinPut: Arg1 (the index of the memo location) is given by the 1-byte unsigned integer following.
//    LongBinPut: Same as BinPut, but with a 4-byte unsigned little-endian integer.
//
// Get, BinGet, LongBinGet: Read an object from the memo and push it on the stack.
//    Get: Arg1 (the memo index) is a newline-terminated decimal string.
//    BinGet: Arg1 (the memo index) is a 1-byte unsigned integer.
//    LongBinGet: Arg1 (the memo index) is a 4-byte unsigned little-endian integer.
//
// SetItem, SetItems: Add one or multiple key+value pairs to an existing dict.
//    SetItem: Add a key+value pair to an existing dict.
//        Stack before:  ... pydict key value
//        Stack after:   ... pydict
//    SetItems: Add an arbitrary number of key+value pairs to an existing dict.
//        Stack before:  ... pydict markobject key_1 value_1 ... key_n value_n
//        Stack after:   ... pydict
//
// NewObj: Build an object instance.
//    The stack before should be thought of as containing a class object followed by an argument tuple
//

struct PickleOpcode
{
    const static uint8_t Mark           = '('; //(0x28, 0P). Push special mark on the stack
    const static uint8_t EmptyTuple     = ')'; //(0x29, 0P). Push empty tuple
    const static uint8_t Stop           = '.'; //(0x2E, 0P). Stop
    const static uint8_t EmptyList      = ']'; //(0x5D, 0P). Push empty list
    const static uint8_t EmptyDict      = '}'; //(0x7D, 0P). Push empty dict
    const static uint8_t Pop            = '0'; //discard topmost stack item
    const static uint8_t PopMark        = '1'; //discard stack top through topmost markobject
    const static uint8_t Dup            = '2'; //duplicate top stack item
    const static uint8_t DecFloat       = 'F'; //push float object; decimal string argument
    const static uint8_t BinFloat       = 'G'; //push float; arg is 8-byte float encoding
    const static uint8_t DecInt         = 'I'; //push integer or bool; decimal string argument
    const static uint8_t BinInt         = 'J'; //(0x4A, 1P), push four-byte signed int
    const static uint8_t BinInt1        = 'K'; //(0x4B, 1P), push 1-byte unsigned int
    const static uint8_t DecLong        = 'L'; //push long; decimal string argument
    const static uint8_t BinInt2        = 'M'; //(0x4D, 1P). Push 2-byte unsigned int
    const static uint8_t None           = 'N'; //(0x4E). Push None
    const static uint8_t Persid         = 'P'; //(0x50, 1P), push persistent object (the argument is a newline-terminated string)
    const static uint8_t BinPersid      = 'Q'; //(0x51, 0P), push an object identified by a persistent ID.
    const static uint8_t Reduce         = 'R'; //(0x52, 0P), push an object built from a callable and an argument tuple.
    const static uint8_t TextString     = 'S'; //push string; NL-terminated string argument
    const static uint8_t BinString      = 'T'; //4-byte length + string, using the encoding given to the reader constructor (default: ascii)
    const static uint8_t ShortBinString = 'U'; //
    const static uint8_t RawUnicode     = 'V'; //push raw unicode string
    const static uint8_t BinUnicode     = 'X'; //0x58, length + string, utf8 encoding
    const static uint8_t Append         = 'a'; //(0x61, 0P). Append stack top to list below it
    const static uint8_t Build          = 'b'; //0x62. Finish building an object, via setstate or dict update (calling __setstate__ or __dict__.update)
    const static uint8_t Global         = 'c'; //0x63. Push self.find_class(modname, name); 2 string args
    const static uint8_t Dict           = 'd'; //build a dict from stack items
    const static uint8_t Appends        = 'e'; //(0x65, 0P). Extend list on stack by topmost stack slice
    const static uint8_t Get            = 'g'; //(0x67, decimal string as memo index). Read an object from the memo and push it on the stack.
    const static uint8_t BinGet         = 'h'; //(0x68, 1-byte memo index). Read an object from the memo and push it on the stack.
    const static uint8_t Inst           = 'i'; //(0x69, ). Build & push class instance
    const static uint8_t LongBinGet     = 'j'; //(0x6A, 4-byte memo index). push item from memo on stack; index is 4-byte arg
    const static uint8_t List           = 'l'; //(0x6C, 0P). Build list from topmost stack items
    const static uint8_t Obj            = 'o'; //build & push class instance
    const static uint8_t Put            = 'p'; //(0x70, 1P), store stack top into the memo
    const static uint8_t BinPut         = 'q'; //(0x71, 1-byte arg), store the stack top into the memo.
    const static uint8_t LongBinPut     = 'r'; //(0x72, 4-byte arg), store the stack top into the memo.
    const static uint8_t SetItem        = 's'; //(0x73, 0P). Add a key+value pair to an existing dict.
    const static uint8_t Tuple          = 't'; //(0x74, 0P). Build tuple from topmost stack items
    const static uint8_t SetItems       = 'u'; //(0x75, 0P). Add all the key+value pairs following the topmost mark object to an existing dict.

    //protocol 2

    const static uint8_t Proto          = 0x80; //identify pickle protocol
    const static uint8_t NewObj         = 0x81; //Build an object instance.
    const static uint8_t Ext1           = 0x82; //push object from extension registry; 1-byte index
    const static uint8_t Ext2           = 0x83; //push object from extension registry; 2-byte index
    const static uint8_t Ext4           = 0x84; //push object from extension registry; 4-byte index
    const static uint8_t Tuple1         = 0x85; //tuple1
    const static uint8_t Tuple2         = 0x86; //tuple2
    const static uint8_t Tuple3         = 0x87; //tuple3
    const static uint8_t NewTrue        = 0x88; //0P. Push True onto the stack
    const static uint8_t NewFalse       = 0x89; //0P. Push False onto the stack
    const static uint8_t Long1          = 0x8A; //push long from < 256 bytes
    const static uint8_t Long4          = 0x8B; //push really big long
};

//cat: category
enum class PickleOpcat
{
    Unknown = 0,
    Proto = 1,
    Stop = 2,
    EmptyCollection, //EmptyList, EmptyTuple, EmptyDict
    PushInt,    //DecInt, BinInt, BinInt1, BinInt2
    PushLong,   //Long1, Long4
    PushFloat,  //DecFloat, BinFloat
    PushString, //TextString, BinString, ShortBinString, RawUnicode, BinUnicode
    PushBool,   //NewTrue, NewFalse
    PushNone,   //None
    Global,
    TupleX,     //Tuple1, Tuple2, Tuple3
    Mark,
    Put,        //Put, BinPut, LongBinPut
    Get,        //Get, BinGet, LongBinGet
    Pop,        //Pop, PopMark
    Dup,
    Append,     //Append, Appends
    SetItems,   //SetItem, SetItems
    Persid,     //Persid, BinPersid
    NewObj,
    Reduce,
    Build
};

//An argument of a pickle operation
struct PickleOparg
{
public:
    enum class DataType
    {
        Int = 0, Float, Str
    };

public:
    DataType t = DataType::Int;
    int nv = 0; //int value
    float fv; //float value
    const char *str = nullptr;

public:
    void SetInt(int v);
    void SetInt(uint8_t v);
    void SetInt(uint16_t v);
    void SetFloat(float v);
    void SetString(const char *v);

    void Print(ostream &strm) const;
};

struct PickleOperation
{
public:
    uint8_t code = (uint8_t)0;
    PickleOpcat cat = PickleOpcat::Unknown;
    int arg_num = 0; //number of arguments
    PickleOparg arg1, arg2;

public:
    void Clear();
};

class PickleReader
{
public:
    const static int SECTION_HEADER_KEY_LEN = 16;
    const static int SECTION_END_KEY_LEN = 4;

public:
    PickleReader();
    virtual ~PickleReader();
    void Clear();

    bool Init();

    bool Read(vector<PickleOperation> &opr_list, IBinaryStream &strm, int stream_id = 0);

    bool ReadSectionHeader(string &section_name, IBinaryStream &strm, bool find_first);

    bool SeekByKey(IBinaryStream &strm, const char *key, int key_len);

    string OpcodeToName(int opcode) const;
    string OpcatToName(PickleOpcat cat) const;

protected:
    StringBlockedHeap str_heap_;
    map<int, string> opcode_map_;
    map<PickleOpcat, string> opcat_map_;

    char section_header_key_[SECTION_HEADER_KEY_LEN];
    char section_end_key_[SECTION_END_KEY_LEN];

protected:
    void BuildOpcodeMap();
    void BuildOpcatMap();

    bool ReadOperation(PickleOperation &opr, IBinaryStream &strm, int stream_id);

    static PickleOpcat GetOpcat(int opcode);

    static bool ReadPutGetArg(int &memo_loc, IBinaryStream &strm, int opcode);

    static bool ReadInt(int &num, IBinaryStream &strm, int opcode);
    static bool ReadLong(int64_t &num, IBinaryStream &strm, int opcode);

    bool ReadStringArg(PickleOparg &arg, IBinaryStream &strm, int opcode);

    bool ReadGlobalArgs(PickleOperation &opr, IBinaryStream &strm);
    bool ReadPersidArgs(PickleOperation &opr, IBinaryStream &strm, int opcode);
};

INFER_FLOW_END

