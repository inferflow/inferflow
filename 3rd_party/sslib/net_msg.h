#pragma once

#include "prime_types.h"
#include "socket_serializable.h"
#include "chained_str_stream.h"
#include <string>

namespace sslib
{

class ChainedNetPacket : public ISocketSerializable
{
public:
    ChainedStrStream str_stream;

public:
    virtual bool Clear() {
        return str_stream.Clear();
    }
    virtual bool Read(Socket *sock);
	virtual bool Write(Socket *sock) const;
};

class NetMsg
{
public:
    uint32_t type = 0, id = 0;
    uint32_t flag = 0;
    std::string contents;

public:
    NetMsg();
    virtual ~NetMsg();

    void Clear();

    virtual bool Read(IBinaryStream &stream);
	virtual bool Write(IBinaryStream &stream) const;
};

struct NaiveNetMsg
{
    uint64_t id = 0;
    std::string str;

    Socket::RetCode Write(Socket &sock, uint32_t magic_code,
        int timeout = INT32_MAX) const;
    static Socket::RetCode Write(Socket &sock, uint32_t magic_code,
        uint64_t id, const std::string &str,
        int timeout = INT32_MAX);
};

} //end of namespace
