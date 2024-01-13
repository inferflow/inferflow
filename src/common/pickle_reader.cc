#include "pickle_reader.h"
#include <sstream>
#include "sslib/log.h"
#include "sslib/stream_helper.h"

INFER_FLOW_BEGIN

using namespace std;
using namespace sslib;

void PickleOparg::SetInt(int v)
{
    this->t = DataType::Int;
    this->nv = v;
}

void PickleOparg::SetInt(uint8_t v)
{
    this->t = DataType::Int;
    this->nv = (int)v;
}

void PickleOparg::SetInt(uint16_t v)
{
    this->t = DataType::Int;
    this->nv = (int)v;
}

void PickleOparg::SetFloat(float v)
{
    this->t = DataType::Float;
    this->fv = v;
}

void PickleOparg::SetString(const char *v)
{
    this->t = DataType::Str;
    this->str = v;
}

void PickleOparg::Print(ostream &strm) const
{
    switch (this->t)
    {
    case DataType::Int:
        strm << this->nv;
        break;
    case DataType::Float:
        strm << this->fv;
        break;
    case DataType::Str:
        strm << this->str;
        break;
    default: break;
    }
}

void PickleOperation::Clear()
{
    code = (uint8_t)0;
    cat = PickleOpcat::Unknown;
    arg_num = 0;
}

////////////////////////////////////////////////////////////////////////////////
// class PickleReader

PickleReader::PickleReader()
{
}

PickleReader::~PickleReader()
{
    Clear();
}

void PickleReader::Clear()
{
    str_heap_.Clear();
}

bool PickleReader::Init()
{
    BuildOpcodeMap();
    BuildOpcatMap();

    char key[] =
    {
        (char)0x50, (char)0x4B, (char)0x03, (char)0x04,
        (char)0x00, (char)0x00, (char)0x08, (char)0x08,
        (char)0x00, (char)0x00, (char)0x00, (char)0x00,
        (char)0x00, (char)0x00, (char)0x00, (char)0x00
    };
    memcpy(section_header_key_, key, SECTION_HEADER_KEY_LEN);

    char end_key[] =
    {
        (char)0x50, (char)0x4B, (char)0x07, (char)0x08
    };
    memcpy(section_end_key_, end_key, SECTION_END_KEY_LEN);
    return true;
}

bool PickleReader::Read(vector<PickleOperation> &opr_list,
    IBinaryStream &strm, int stream_id)
{
    bool ret = true;
    opr_list.clear();

    int protocol_version = 2;
    PickleOperation opr;

    bool is_done = false;
    while (ret && !is_done)
    {
        ret = ReadOperation(opr, strm, stream_id);
        if (!ret) {
            return false;
        }

        opr_list.push_back(opr);

        if (opr.code == PickleOpcode::Proto) {
            protocol_version = opr.arg1.nv;
        }

        if (opr.code == PickleOpcode::Stop)
        {
            uint64_t offset = strm.TellRd();
            char buf[SECTION_END_KEY_LEN + 1];
            ret = strm.Read(buf, SECTION_END_KEY_LEN);
            strm.SeekRd(offset);

            if (memcmp(buf, section_end_key_, SECTION_END_KEY_LEN) == 0
                || buf[0] == '\0')
            {
                is_done = true;
            }
        }
    }

    (void)protocol_version;
    //LogKeyInfo("protocol_version: %d", protocol_version);
    return ret;
}

bool PickleReader::ReadSectionHeader(string &section_name,
    IBinaryStream &strm, bool find_first)
{
    bool ret = true;
    section_name.clear();

    uint64_t offset = strm.TellRd();
    if (find_first)
    {
        ret = SeekByKey(strm, section_header_key_, SECTION_HEADER_KEY_LEN);
        Macro_RetxFalseIf(!ret, LogError("Failed to find the next tensor data section"));

        offset = strm.TellRd();
        ret = strm.SeekRd(offset + 10);
    }
    else
    {
        char buf[SECTION_HEADER_KEY_LEN + 1];
        strm.Read(buf, SECTION_HEADER_KEY_LEN);
        if (memcmp(buf, section_header_key_, SECTION_HEADER_KEY_LEN) == 0)
        {
            ret = strm.SeekRd(offset + SECTION_HEADER_KEY_LEN + 10);
        }
        else
        {
            strm.SeekRd(offset);
            return true;
        }
    }

    uint16_t len1 = 0, len2 = 0;
    ret = ret && strm.Read(len1);
    ret = ret && strm.Read(len2);
    //LogKeyInfo("len1: %d; len2: %d", (int)len1, (int)len2);
    if (ret && len1 >= 255)
    {
        LogError("len1 is too large: %u", len1);
        return false;
    }

    char section_name_buf[255 + 1];
    ret = ret && strm.Read(section_name_buf, len1);
    section_name_buf[len1] = '\0';
    section_name = section_name_buf;

    if (ret)
    {
        offset = strm.TellRd();
        ret = strm.SeekRd(offset + len2);
    }

    return ret;
}

bool PickleReader::SeekByKey(IBinaryStream &strm, const char *key, int key_len)
{
    bool ret = true;
    if (key_len <= 0) {
        return false;
    }

    char ch0 = key[0];
    char cur_char = ' ';
    while (strm.Read(cur_char))
    {
        if (cur_char != ch0) {
            continue;
        }

        uint64_t pos_bak = strm.TellRd();
        bool is_same = true;
        for (int idx = 1; idx < key_len; idx++)
        {
            strm.Read(cur_char);
            if (cur_char != key[idx])
            {
                is_same = false;
                break;
            }
        }
        if (is_same) {
            return true;
        }

        strm.SeekRd(pos_bak);
    }

    return ret;
}

string PickleReader::OpcodeToName(int opcode) const
{
    auto iter = opcode_map_.find(opcode);
    return iter != opcode_map_.end() ? iter->second : "Unknown";
}

string PickleReader::OpcatToName(PickleOpcat cat) const
{
    auto iter = opcat_map_.find(cat);
    return iter != opcat_map_.end() ? iter->second : "Unknown";
}

bool PickleReader::ReadOperation(PickleOperation &opr, IBinaryStream &strm, int stream_id)
{
    opr.Clear();
    uint64_t read_offset = strm.TellRd();
    bool ret = strm.Read(opr.code);
    if (!ret) {
        LogError("Error occurred in reading the opcode");
        return false;
    }

    opr.cat = GetOpcat(opr.code);

    uint8_t u8 = (uint8_t)0;
    int num = 0;
    int64_t n64 = 0;
    switch (opr.cat)
    {
    case PickleOpcat::Proto:
        ret = strm.Read(u8);
        opr.arg_num = 1;
        opr.arg1.SetInt(u8);
        break;
    case PickleOpcat::Stop:
        //no action is needed (arg_num has been set to 0 before)
        break;
    case PickleOpcat::EmptyCollection:
        //no arguments
        break;
    case PickleOpcat::Mark:
        //no arguments
        break;
    case PickleOpcat::Put:
        ReadPutGetArg(num, strm, opr.code);
        opr.arg_num = 1;
        opr.arg1.SetInt(num);
        break;
    case PickleOpcat::Get:
        ReadPutGetArg(num, strm, opr.code);
        opr.arg_num = 1;
        opr.arg1.SetInt(num);
        break;
    case PickleOpcat::PushInt:
        ret = ReadInt(num, strm, opr.code);
        opr.arg_num = 1;
        opr.arg1.SetInt(num);
        break;
    case PickleOpcat::PushLong:
        ret = ReadLong(n64, strm, opr.code);
        opr.arg_num = 1;
        //opr.arg1.SetLong(n64);
        break;
    case PickleOpcat::PushString:
        ret = ReadStringArg(opr.arg1, strm, opr.code);
        opr.arg_num = 1;
        break;
    case PickleOpcat::PushBool:
        //no arguments
        break;
    case PickleOpcat::PushNone:
        //no arguments
        break;
    case PickleOpcat::Global:
        ret = ReadGlobalArgs(opr, strm);
        opr.arg_num = 2;
        break;
    case PickleOpcat::TupleX:
        //no arguments
        break;
    case PickleOpcat::SetItems:
        //no arguments
        break;
    case PickleOpcat::NewObj:
        //no arguments
        break;
    case PickleOpcat::Append:
        //no arguments
        break;
    case PickleOpcat::Reduce:
        //no arguments
        break;
    case PickleOpcat::Persid:
        ret = ReadPersidArgs(opr, strm, opr.code);
        opr.arg_num = opr.code == PickleOpcode::Persid ? 1 : 0;
        break;
    case PickleOpcat::Build:
        //no arguments
        break;
    default:
        LogError("Invalid opcode 0x%X in stream %d", (int)opr.code, stream_id);
        LogKeyInfo("Offset: %I64u. Bytes after the opcode:", read_offset);
        {
            stringstream ss;
            char buf[32], hex_buf[16];
            bool is_succ = strm.Read(buf, 32);
            for (int idx = 0; is_succ && idx < 32; idx++)
            {
                sprintf(hex_buf, "0x%X", (int)(uint8_t)buf[idx]);
                ss << (idx > 0 ? ", " : "  ") << (int)(uint8_t)buf[idx]
                    << " (" << hex_buf << ")";
            }
            LogKeyInfo("%s", ss.str().c_str());
        }
        return false;
    }

    return ret;
}

void PickleReader::BuildOpcodeMap()
{
    opcode_map_.clear();
    auto &opcode_map = opcode_map_;

    opcode_map[PickleOpcode::Mark] = "Mark";
    opcode_map[PickleOpcode::EmptyTuple] = "EmptyTuple";
    opcode_map[PickleOpcode::Stop] = "Stop";
    opcode_map[PickleOpcode::EmptyList] = "EmptyList";
    opcode_map[PickleOpcode::EmptyDict] = "EmptyDict";
    opcode_map[PickleOpcode::Pop] = "Pop";
    opcode_map[PickleOpcode::PopMark] = "PopMark";
    opcode_map[PickleOpcode::Dup] = "Dup";
    opcode_map[PickleOpcode::DecFloat] = "DecFloat";
    opcode_map[PickleOpcode::BinFloat] = "BinFloat";
    opcode_map[PickleOpcode::DecInt] = "DecInt";
    opcode_map[PickleOpcode::BinInt] = "BinInt";
    opcode_map[PickleOpcode::BinInt1] = "BinInt1";
    opcode_map[PickleOpcode::DecLong] = "DecLong";
    opcode_map[PickleOpcode::BinInt2] = "BinInt2";
    opcode_map[PickleOpcode::None] = "None";
    opcode_map[PickleOpcode::Persid] = "Persid";
    opcode_map[PickleOpcode::BinPersid] = "BinPersid";
    opcode_map[PickleOpcode::Reduce] = "Reduce";
    opcode_map[PickleOpcode::TextString] = "TextString";
    opcode_map[PickleOpcode::BinString] = "BinString";
    opcode_map[PickleOpcode::ShortBinString] = "ShortBinString";
    opcode_map[PickleOpcode::RawUnicode] = "RawUnicode";
    opcode_map[PickleOpcode::BinUnicode] = "BinUnicode";
    opcode_map[PickleOpcode::Append] = "Append";
    opcode_map[PickleOpcode::Build] = "Build";
    opcode_map[PickleOpcode::Global] = "Global";
    opcode_map[PickleOpcode::Dict] = "Dict";
    opcode_map[PickleOpcode::Appends] = "Appends";
    opcode_map[PickleOpcode::Get] = "Get";
    opcode_map[PickleOpcode::BinGet] = "BinGet";
    opcode_map[PickleOpcode::Inst] = "Inst";
    opcode_map[PickleOpcode::LongBinGet] = "LongBinGet";
    opcode_map[PickleOpcode::List] = "List";
    opcode_map[PickleOpcode::Obj] = "Obj";
    opcode_map[PickleOpcode::Put] = "Put";
    opcode_map[PickleOpcode::BinPut] = "BinPut";
    opcode_map[PickleOpcode::LongBinPut] = "LongBinPut";
    opcode_map[PickleOpcode::SetItem] = "SetItem";
    opcode_map[PickleOpcode::Tuple] = "Tuple";
    opcode_map[PickleOpcode::SetItems] = "SetItems";

    opcode_map[PickleOpcode::Proto] = "Proto";
    opcode_map[PickleOpcode::NewObj] = "NewObj";
    opcode_map[PickleOpcode::Ext1] = "Ext1";
    opcode_map[PickleOpcode::Ext2] = "Ext2";
    opcode_map[PickleOpcode::Ext4] = "Ext4";
    opcode_map[PickleOpcode::Tuple1] = "Tuple1";
    opcode_map[PickleOpcode::Tuple2] = "Tuple2";
    opcode_map[PickleOpcode::Tuple3] = "Tuple3";
    opcode_map[PickleOpcode::NewTrue] = "NewTrue";
    opcode_map[PickleOpcode::NewFalse] = "NewFalse";
    opcode_map[PickleOpcode::Long1] = "Long1";
    opcode_map[PickleOpcode::Long4] = "Long4";
}

void PickleReader::BuildOpcatMap()
{
    opcat_map_.clear();
    auto &opcat_map = opcat_map_;

    opcat_map[PickleOpcat::Proto] = "Proto";
    opcat_map[PickleOpcat::Stop] = "Stop";
    opcat_map[PickleOpcat::EmptyCollection] = "EmptyCollection";
    opcat_map[PickleOpcat::PushInt] = "PushInt";
    opcat_map[PickleOpcat::PushLong] = "PushLong";
    opcat_map[PickleOpcat::PushFloat] = "PushFloat";
    opcat_map[PickleOpcat::PushString] = "PushString";
    opcat_map[PickleOpcat::PushBool] = "PushBool";
    opcat_map[PickleOpcat::PushNone] = "PushNone";
    opcat_map[PickleOpcat::Global] = "Global";
    opcat_map[PickleOpcat::TupleX] = "TupleX";
    opcat_map[PickleOpcat::Mark] = "Mark";
    opcat_map[PickleOpcat::Put] = "Put";
    opcat_map[PickleOpcat::Get] = "Get";
    opcat_map[PickleOpcat::Pop] = "Pop";
    opcat_map[PickleOpcat::Dup] = "Dup";
    opcat_map[PickleOpcat::Append] = "Append";
    opcat_map[PickleOpcat::SetItems] = "SetItems";
    opcat_map[PickleOpcat::Persid] = "Persid";
    opcat_map[PickleOpcat::NewObj] = "NewObj";
    opcat_map[PickleOpcat::Reduce] = "Reduce";
    opcat_map[PickleOpcat::Build] = "Build";
}

//static
PickleOpcat PickleReader::GetOpcat(int opcode)
{
    PickleOpcat cat = PickleOpcat::Unknown;
    switch (opcode)
    {
    case PickleOpcode::Proto:
        cat = PickleOpcat::Proto;
        break;
    case PickleOpcode::Stop:
        cat = PickleOpcat::Stop;
        break;
    case PickleOpcode::Mark:
        cat = PickleOpcat::Mark;
        break;
    case PickleOpcode::EmptyTuple:
    case PickleOpcode::EmptyList:
    case PickleOpcode::EmptyDict:
        cat = PickleOpcat::EmptyCollection;
        break;
    case PickleOpcode::Put:
    case PickleOpcode::BinPut:
    case PickleOpcode::LongBinPut:
        cat = PickleOpcat::Put;
        break;
    case PickleOpcode::Get:
    case PickleOpcode::BinGet:
    case PickleOpcode::LongBinGet:
        cat = PickleOpcat::Get;
        break;
    case PickleOpcode::Pop:
    case PickleOpcode::PopMark:
        cat = PickleOpcat::Pop;
        break;
    case PickleOpcode::Dup:
        cat = PickleOpcat::Dup;
        break;
    case PickleOpcode::DecInt:
    case PickleOpcode::BinInt:
    case PickleOpcode::BinInt1:
    case PickleOpcode::BinInt2:
        cat = PickleOpcat::PushInt;
        break;
    case PickleOpcode::Long1:
    case PickleOpcode::Long4:
        cat = PickleOpcat::PushLong;
        break;
    case PickleOpcode::DecFloat:
    case PickleOpcode::BinFloat:
        cat = PickleOpcat::PushFloat;
        break;
    case PickleOpcode::TextString:
    case PickleOpcode::BinString:
    case PickleOpcode::ShortBinString:
    case PickleOpcode::RawUnicode:
    case PickleOpcode::BinUnicode:
        cat = PickleOpcat::PushString;
        break;
    case PickleOpcode::NewTrue:
    case PickleOpcode::NewFalse:
        cat = PickleOpcat::PushBool;
        break;
    case PickleOpcode::None:
        cat = PickleOpcat::PushNone;
        break;
    case PickleOpcode::Global:
        cat = PickleOpcat::Global;
        break;
    case PickleOpcode::Tuple:
    case PickleOpcode::Tuple1:
    case PickleOpcode::Tuple2:
    case PickleOpcode::Tuple3:
        cat = PickleOpcat::TupleX;
        break;
    case PickleOpcode::SetItem:
    case PickleOpcode::SetItems:
        cat = PickleOpcat::SetItems;
        break;
    case PickleOpcode::NewObj:
        cat = PickleOpcat::NewObj;
        break;
    case PickleOpcode::Append:
    case PickleOpcode::Appends:
        cat = PickleOpcat::Append;
        break;
    case PickleOpcode::Reduce:
        cat = PickleOpcat::Reduce;
        break;
    case PickleOpcode::Persid:
    case PickleOpcode::BinPersid:
        cat = PickleOpcat::Persid;
        break;
    case PickleOpcode::Build:
        cat = PickleOpcat::Build;
        break;
    default: break;
    }

    return cat;
}

//static
bool PickleReader::ReadPutGetArg(int &memo_loc, IBinaryStream &strm, int opcode)
{
    bool ret = false; //default false
    memo_loc = 0;

    uint8_t v8 = (uint8_t)0;
    switch (opcode)
    {
    case PickleOpcode::BinPut:
    case PickleOpcode::BinGet:
        ret = strm.Read(v8);
        memo_loc = (int)v8;
        break;
    case PickleOpcode::LongBinPut:
    case PickleOpcode::LongBinGet:
        ret = strm.Read(memo_loc);
        break;
    case PickleOpcode::Put:
    case PickleOpcode::Get:
        {
            string str;
            ret = strm.GetLine(str);
            memo_loc = atoi(str.c_str());
        }
        break;
    default:
        break;
    }

    return ret;
}

//static
bool PickleReader::ReadInt(int &num, IBinaryStream &strm, int opcode)
{
    num = 0;
    uint8_t v8 = 0;
    uint16_t v16 = 0;
    bool ret = false; //default false
    switch (opcode)
    {
    case PickleOpcode::BinInt:
        ret = strm.Read(num);
        break;
    case PickleOpcode::BinInt1:
        ret = strm.Read(v8);
        if (ret) {
            num = (int)v8;
        }
        break;
    case PickleOpcode::BinInt2:
        ret = strm.Read(v16);
        if (ret) {
            num = (int)v16;
        }
        break;
    default:
        break;
    }

    return ret;
}

//static
bool PickleReader::ReadLong(int64_t &num, IBinaryStream &strm, int opcode)
{
    num = 0;
    char buf[256];
    uint8_t v8 = 0;
    uint32_t v32 = 0;
    bool ret = false; //default false
    switch (opcode)
    {
    case PickleOpcode::Long1:
        ret = strm.Read(v8);
        ret = ret && strm.Read(buf, v8);
        break;
    case PickleOpcode::Long4:
        ret = strm.Read(v32);
        LogError("Long4 is not handled yet");
        ret = false;
        break;
    default:
        break;
    }

    return ret;
}

bool PickleReader::ReadStringArg(PickleOparg &arg, IBinaryStream &strm, int opcode)
{
    bool ret = false; //default false
    //int n32 = 0;
    //uint8_t n8 = (uint8_t)0;
    string str;

    switch (opcode)
    {
    case PickleOpcode::BinUnicode:
        ret = BinStreamHelper::ReadString32(strm, str);
        break;
    default:
        break;
    }

    if (ret) {
        arg.SetString(str_heap_.AddString(str));
    }
    return ret;
}

bool PickleReader::ReadGlobalArgs(PickleOperation &opr, IBinaryStream &strm)
{
    string str;
    bool ret = strm.GetLine(str);
    if (ret) {
        opr.arg1.SetString(str_heap_.AddString(str));
    }

    ret = ret && strm.GetLine(str);
    if (ret) {
        opr.arg2.SetString(str_heap_.AddString(str));
    }

    return ret;
}

bool PickleReader::ReadPersidArgs(PickleOperation &opr, IBinaryStream &strm, int opcode)
{
    bool ret = false; //default false
    string str;
    switch (opcode)
    {
    case PickleOpcode::Persid:
        ret = strm.GetLine(str);
        if (ret) {
            opr.arg1.SetString(str_heap_.AddString(str));
        }
        break;
    case PickleOpcode::BinPersid:
        ret = true;
        break;
    default:
        break;
    }

    return ret;
}

INFER_FLOW_END

