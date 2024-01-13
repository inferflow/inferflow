#include "int_dict.h"
#include "log.h"
#include "binary_file_stream.h"

using namespace std;

namespace sslib
{

IntDict::IntDict() : id2idx_map_(0, 1.5f, true)
{
}

IntDict::~IntDict()
{
    Clear();
}

uint32_t IntDict::Id2Index(uint32_t item_id) const
{
    HashMapUInt32::ConstIterator iter = id2idx_map_.Find(item_id);
    return iter.IsEnd() ? UINT32_MAX : iter.Value();
}

uint32_t IntDict::Index2Id(uint32_t item_idx) const
{
    return item_idx < (uint32_t)items_.Size() ? items_[item_idx] : UINT32_MAX;
}

void IntDict::Clear()
{
    items_.Clear();
    id2idx_map_.Clear();
}

uint32_t IntDict::AddItem(uint32_t id, uint32_t idx)
{
    auto iter = id2idx_map_.Find(id);
    if(!iter.IsEnd()) {
        return iter.Value();
    }

    uint32_t current_size = (uint32_t)items_.Size();
    if(idx == UINT32_MAX) {
        idx = current_size;
    }

    if(idx >= UINT32_MAX) {
        return UINT32_MAX;
    }

    if (idx >= current_size)
    {
        items_.Resize(idx + 1);
        for (uint32_t iItem = current_size; iItem <= idx; iItem++) {
            items_[iItem] = UINT32_MAX;
        }
    }

    items_[idx] = id;
    id2idx_map_.Insert(id, idx);
    return idx;
}

bool IntDict::Store(const std::string &path) const
{
    BinFileStream stream;
    bool ret = stream.Open(path.c_str(), BinFileStream::MODE_WRITE|BinFileStream::MODE_CREATE|BinFileStream::MODE_TRUNC);
    if(!ret) {
        LogError("Failed to open file %s", path.c_str());
        return false;
    }

    ret = Write(stream);
    ret = ret && stream.Flush();
    return ret && stream.IsGood();
}

bool IntDict::Load(const std::string &path)
{
    BinFileStream stream;
    bool ret = stream.Open(path.c_str());
    if(!ret) {
        LogError("Failed to open file %s", path.c_str());
        return false;
    }

    return Read(stream);
}

bool IntDict::Print(const std::string &path, uint32_t options) const
{
    ofstream ps(path.c_str());
    if(!ps) {
        LogWarning("Failed to open file %s for printing the IntDict", path.c_str());
        return false;
    }

    uint32_t item_num = GetItemNum();
    ps << "item_num: " << item_num << endl;

    if((options & PrintOpt_SortByIdx) != 0)
    {
        for(uint32_t idx = 0; idx < item_num; idx++)
        {
            ps << idx << "\t" << Index2Id(idx) << endl;
        }
    }
    else
    {
        LogWarning("Not implemented yet");
    }

    ps.flush();
    return ps.good();
}

bool IntDict::Write(IBinaryStream &stream) const
{
    uint32_t item_num = GetItemNum();
    bool ret = stream.Write((const char*)&item_num, sizeof(item_num));

    uint32_t id = 0;
    for(uint32_t iItem = 0; ret && iItem < item_num; ++iItem)
    {
        id = items_[iItem];
        ret = stream.Write((const char*)&id, sizeof(id));
    }

    return ret && stream.IsGood();
}

bool IntDict::Read(IBinaryStream &stream)
{
    Clear();
    uint32_t item_num = 0;
    bool ret = stream.Read((char*)&item_num, sizeof(item_num));

    uint32_t id = 0;
    for(uint32_t iItem = 0; ret && iItem < item_num; iItem++)
    {
        ret = stream.Read((char*)&id, sizeof(id));
        AddItem(id, iItem);
    }

    ret = ret && stream.IsGood();
    return ret;
}

} //end of namespace
