#include "net_msg.h"
#include "stream_helper.h"
#include <chrono>

using namespace std;

namespace sslib
{

bool ChainedNetPacket::Read(Socket *sock)
{
    bool ret = true;
    str_stream.Clear();

    uint32_t block_num = 0;
    ret = sock->Readn((char*)&block_num, sizeof(block_num)) > 0;

    uint32_t block_size = 0;
    ChainedStrStream::Block block;
    for (uint32_t block_idx = 0; ret && block_idx < block_num; block_idx++)
    {
        ret = sock->Readn((char*)&block_size, sizeof(block_size)) > 0;
        if (!ret) {
            break;
        }

        block.size = block.capacity = block_size;
        block.content = new char[block_size];
        ret = sock->Readn((char*)block.content, block_size) > 0;
        if (ret) {
            ret = str_stream.AddBlock(block);
        }

        if (!ret) {
            delete[] block.content;
            break;
        }
    }

    return ret;
}

bool ChainedNetPacket::Write(Socket *sock) const
{
    bool ret = true;

    uint32_t block_num = (uint32_t)str_stream.GetBlockNum();
    sock->Writen((const char*)&block_num, sizeof(block_num));

    ChainedStrStream::Block block;
    uint32_t block_size = 0;
    for (uint32_t block_idx = 0; ret && block_idx < block_num; block_idx++)
    {
        ret = str_stream.GetBlock(block_idx, block);
        if (ret && block.size > 0)
        {
            block_size = (uint32_t)block.size;
            ret = sock->Writen((const char*)&block_size, sizeof(block_size)) > 0;
            if (ret) {
                ret = sock->Writen(block.content, (int)block.size) > 0;
            }
        }
    }

    return ret;
}

///////////////////////////////////////////////////////////////////////////////
// class NetMsg

NetMsg::NetMsg()
{
    type = 0;
    id = 0;
    flag = 0;
    contents = "";
}

NetMsg::~NetMsg()
{
    Clear();
}

void NetMsg::Clear()
{
    type = 0;
    id = 0;
    flag = 0;
    contents = "";
}

bool NetMsg::Read(IBinaryStream &stream)
{
    bool ret = true;
    ret = stream.Read((char*)&type, sizeof(type));
    ret = ret && stream.Read((char*)&id, sizeof(id));
    ret = ret && stream.Read((char*)&flag, sizeof(flag));
    ret = ret && BinStreamHelper::ReadString32(stream, contents);
    return ret;
}

bool NetMsg::Write(IBinaryStream &stream) const
{
    bool ret = true;
    ret = stream.Write((const char*)&type, sizeof(type));
    ret = ret && stream.Write((const char*)&id, sizeof(id));
    ret = ret && stream.Write((const char*)&flag, sizeof(flag));
    ret = ret && BinStreamHelper::WriteString32(stream, contents);
    return ret;
}

Socket::RetCode NaiveNetMsg::Write(Socket &sock,
    uint32_t magic_code, int timeout) const
{
    return Write(sock, magic_code, id, str, timeout);
}

//static
Socket::RetCode NaiveNetMsg::Write(Socket &sock, uint32_t magic_code, uint64_t id,
    const string &str, int timeout)
{
    (void)timeout;
    uint32_t length = (uint32_t)str.size();
    if (length > (uint32_t)INT32_MAX)
    {
        LogError("Too large message");
        return Socket::RetCode::InvalidData;
    }

    const uint32_t buf_len = 4 + 8 + 4;
    char buf[buf_len];
    memcpy(buf, &magic_code, 4); //4: sizeof(magic_code)
    memcpy(buf + 4, &id, 8); //8: sizeof(id)
    memcpy(buf + 12, &length, 4); //4: sizeof(length)

    //auto start_tm = chrono::steady_clock::now();
    Socket::RetCode ret_code = sock.WriteAll(buf, buf_len);
    if (ret_code != Socket::RetCode::Success) {
        return ret_code;
    }

    //auto now_tm = chrono::steady_clock::now();
    //int elapsed_time = (int)chrono::duration_cast<chrono::milliseconds>(now_tm - start_tm).count();
    //if (elapsed_time > timeout) {
    //    return false;
    //}

    //bytes_written = sock.Writen(str.c_str(), length, timeout - elapsed_time);
    ret_code = sock.WriteAll(str.c_str(), length);
    return ret_code;
}

} //end of namespace
