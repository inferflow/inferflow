#pragma once

#include "socket.h"
#include "number.h"
#include "string.h"
#include "http_common.h"
#include <map>
#include <chrono>
#include "cache.h"
#ifdef ENABLE_HTTPS
#   include "openssl/ssl.h"
#   include "openssl/err.h"
#endif //ENABLE_HTTPS

namespace sslib
{

using std::string;
using std::chrono::time_point;
using std::chrono::steady_clock;

class HttpClient;

class HttpResponseReader
{
public:
    HttpRetCode NextChunk(string &chunk_str);

    int status_code() const {
        return res_.status_code;
    }

    const std::vector<string>& header_lines() const {
        return res_.header_lines;
    }

protected:
    HttpResponse res_;
    HttpMessageStream strm_;
    HttpStreamingData streaming_data_;
    int chunk_offset_ = 0;

    friend class HttpClient;
    HttpClient *client_ = nullptr;
};

class HttpClient
{
public:
    HttpClient();
    virtual ~HttpClient();

    bool Init(const string &ca_data = "", bool ca_data_is_file_path = false);

    HttpRetCode Execute(HttpResponse &response, const HttpRequest &request,
        int timeout = INT32_MAX, int connect_timeout = 0,
        uint32_t max_body_length = UINT32_MAX);

    HttpRetCode Execute(HttpResponseReader &res_reader, const HttpRequest &request,
        int timeout = INT32_MAX, int connect_timeout = 0,
        uint32_t max_body_length = UINT32_MAX);

    static const char* RetCodeToString(HttpRetCode ret_code);

protected:
    struct SocketInfo
    {
        Socket socket;
        bool is_connected = false;
        chrono::time_point<chrono::steady_clock> start_time;

        void Close()
        {
            socket.Shutdown();
            socket.Close();
            is_connected = false;
        }
    };

    struct UrlInfo
    {
    public:
        string url;
        string addr; //server address
        int port = 0;
        string addr_and_port;
        string protocol_name;
        string resource_part;

    public:
        void Parse(const string &url_str);
    };

    class HttpResponseCombo
    {
    public:
        HttpResponse *res = nullptr;
        HttpResponseReader *reader = nullptr;

    public:
        HttpResponseCombo(HttpResponse *p_res = nullptr, HttpResponseReader *p_reader = nullptr)
        {
            res = p_res;
            reader = p_reader;
        }
    };

protected:
    LruCache<string, SocketInfo> socket_cache_;

    int buf_len_ = 65535;
    char *buf_ = nullptr;

    std::map<string, HttpHeaderFieldId, StrLessNoCase> header_field_map_;

#ifdef ENABLE_HTTPS
    SSL_CTX *ssl_ctx_ = nullptr;
#endif //ENABLE_HTTPS

protected:
    HttpRetCode ExecuteInner(HttpResponseCombo &res_combo, const HttpRequest &request,
        int timeout = INT32_MAX, int connect_timeout = 0,
        uint32_t max_body_length = UINT32_MAX);

    HttpRetCode SendAndReceive(HttpResponse &response,
        HttpStreamingData *streaming_data,
        const string &request_str,
        SocketInfo &socket_info,
        const time_point<steady_clock> &start_tm,
        int timeout, bool is_proxy,
        uint32_t max_body_length = UINT32_MAX) const;

    HttpRetCode ReceiveResponse(HttpResponse &response,
        HttpStreamingData *streaming_data,
        SocketInfo &socket_info, int timeout,
        bool is_end_upon_empty_line = false,
        uint32_t max_body_length = UINT32_MAX,
        bool clear_response_first = true) const;

    static void BuildRequestString(string &request_str, const HttpRequest &request,
        const UrlInfo &url_info, bool use_full_url);
    static string GetRedirectionServer(const HttpResponse &response);

protected:
#ifdef ENABLE_HTTPS
    struct BioStream
    {
    public:
        BIO *ptr = nullptr;

    public:
        BioStream() {};
        virtual ~BioStream()
        {
            Clear();
        }
        void Clear()
        {
            if (ptr != nullptr)
            {
                BIO_free_all(ptr);
                ptr = nullptr;
            }
        }
    };

    HttpRetCode ReceiveHttpsResponse(HttpResponse &response, BioStream &bio_strm,
        uint32_t max_body_length = UINT32_MAX);
    bool CreateHttpsConnection(BioStream&strm, const UrlInfo &url_info) const;
    bool SetupHttpsConnection(BioStream&strm, const UrlInfo &url_info) const;
    bool SetCertificate(const char *buf, int length);
#endif //ENABLE_HTTPS

private:
    friend class HttpsClient;
    friend class HttpResponseReader;
};

} //end of namespace
