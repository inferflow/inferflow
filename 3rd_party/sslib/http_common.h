#pragma once

#include <vector>
#include <map>
#include "http_utility.h"

namespace sslib
{

using std::string;

class HttpMessage
{
public:
    std::vector<string> header_lines;
    int body_length = 0;
    string body;

public:
    virtual ~HttpMessage();
    void Clear();

    void AddConnectionHeader(bool keep_alive = true);
};

class HttpRequest : public HttpMessage
{
public:
    enum class Method
    {
        Unknown = 0, Get, Head, Post, Put, Delete, CONNECT, OPTIONS, TraceMethod, PATCH
    };

    typedef Method Type;

public:
    Method type = Method::Unknown;
    string url;
    string proxy_url, user_agent;
    string remote_address;
    bool need_streaming_response = false;
};

class HttpResponse : public HttpMessage
{
public:
    int status_code = 0;
    bool keep_alive = false;

public:
    void Clear()
    {
        HttpMessage::Clear();
        status_code = 0;
    }
};

struct HttpStreamingData;

class HttpMessageStream
{
public:
    string str;
    size_t offset = 0;

public:
    enum class UpdatingState
    {
        Header, Body, End
    };

    struct HeaderInfo
    {
        bool keep_alive = false;
        bool is_close = false;
        bool is_chunked_transfer_encoding = false;
        int chunk_length = 0;
    };

public:
    void UpdateMessage(HttpMessage &msg, HeaderInfo &hdr, UpdatingState &state,
        HttpStreamingData *streaming_data, const HttpHeaderFieldMap &header_field_map);
    bool ReadCompleteLine(string &line_str);

    static int GetChunkLength(const string &line_str);
};

struct HttpStreamingData
{
    vector<string> chunks;

    HttpMessageStream::UpdatingState state = HttpMessageStream::UpdatingState::Header;
    HttpMessageStream::HeaderInfo hdr;
    HttpMessageStream *strm = nullptr;
    void *socket_info = nullptr;
};

enum class HttpRetCode
{
    Success = 0,
    InvalidRequest = 1,
    ConnectionError = 2,
    ProxyError = 3,
    SendTimeout = 11, SendDisconnected = 12, SendError = 13,
    RecvTimeout = 21, RecvDisconnected = 22, RecvError = 23,
    OtherError = 99
};

} //end of namespace
