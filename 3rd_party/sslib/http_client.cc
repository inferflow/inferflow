#include "http_client.h"
#include <iostream>
#include "prime_types.h"
#include "string.h"
#include "log.h"
#include "macro.h"
#include <sstream>
#include "thread.h"
#include "path.h"
#include <signal.h>
#include "ca_cert.pem.inc" //define ca_cert_buf

namespace sslib
{

using namespace std;
using namespace std::chrono;

HttpRetCode HttpResponseReader::NextChunk(string &chunk_str)
{
    HttpRetCode ret_code = HttpRetCode::Success;
    chunk_str.clear();

    bool is_end = streaming_data_.state == HttpMessageStream::UpdatingState::End;
    int chunk_num = (int)streaming_data_.chunks.size();

    if (!is_end && chunk_offset_ >= chunk_num)
    {
        if (client_ == nullptr || streaming_data_.socket_info == nullptr)
        {
            //LogWarning("Warning: Null client or socket_info");
            return HttpRetCode::RecvError;
        }

        int timeout = INT32_MAX;
        auto *socket_info = (HttpClient::SocketInfo*)streaming_data_.socket_info;
        ret_code = client_->ReceiveResponse(res_, &streaming_data_, *socket_info,
            timeout, false, UINT32_MAX, false);
        chunk_num = (int)streaming_data_.chunks.size();
    }

    if (chunk_offset_ < chunk_num)
    {
        chunk_str = streaming_data_.chunks[chunk_offset_];
        chunk_offset_++;
        return ret_code;
    }

    if (is_end) {
        return ret_code;
    }

    return ret_code;
}

HttpClient::HttpClient() : socket_cache_(5)
{
    buf_ = new char[buf_len_ + 1];
    memset(buf_, 0, buf_len_ + 1);
    HttpUtility::InitHttpFieldMap(header_field_map_);
}

HttpClient::~HttpClient()
{
    if (buf_ != nullptr)
    {
        delete[] buf_;
        buf_ = nullptr;
    }
}

bool HttpClient::Init(const string &ca_data, bool ca_data_is_file_path)
{
    (void)ca_data; (void)ca_data_is_file_path;
    bool ret = true;

#ifdef ENABLE_HTTPS
    //LogKeyInfo("Will init the SSL lib");
    SSL_library_init();
    //LogKeyInfo("Will load the error strings");
    SSL_load_error_strings();

#   ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#   endif

    //LogKeyInfo("Will get the SSL method");
    const SSL_METHOD *ssl_method = SSLv23_method();
    if (nullptr == ssl_method)
    {
        LogError("Failed to get the SSL method");
        return false;
    }

    ssl_ctx_ = SSL_CTX_new(ssl_method);
    if (ssl_ctx_ == nullptr)
    {
        LogError("Failed to create the SSL context");
        return false;
    }

    //LogKeyInfo("Will setup the SSL context");
    SSL_CTX_set_verify(ssl_ctx_, SSL_VERIFY_PEER, nullptr);
    SSL_CTX_set_verify_depth(ssl_ctx_, 4);

    const long flags = SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_COMPRESSION;
    SSL_CTX_set_options(ssl_ctx_, flags);

    if (!ca_data.empty())
    {
        if (ca_data_is_file_path)
        {
            //vector<string> lines;
            //Path::GetFileContent_Text(ca_data, lines);
            //ofstream out(ca_file_path + ".cc");
            //for (const string &line : lines)
            //{
            //    out << "\"" << line << "\\n\"\n";
            //}
            //out.close();

            string file_content;
            ret = Path::GetFileContent_Text(ca_data, file_content);
            if (ret) {
                ret = SetCertificate(file_content.c_str(), (int)file_content.length());
            }
        }
        else
        {
            //LogKeyInfo("ca_data size: %d", (int)ca_data.size());
            ret = SetCertificate(ca_data.c_str(), (int)ca_data.length());
        }
        Macro_RetxFalseIf(!ret, LogError("Failed to set certificate"));

        //int res = SSL_CTX_load_verify_locations(ssl_ctx_, ca_file_path.c_str(), nullptr);
        //if (1 != res) {
        //    return false;
        //}
    }
    else
    {
        ret = SetCertificate(ca_cert_buf, (int)strlen(ca_cert_buf));
        Macro_RetxFalseIf(!ret, LogError("Failed to set certificate"));
    }
#endif //ENABLE_HTTPS

    return ret;
}

//static
const char* HttpClient::RetCodeToString(HttpRetCode ret_code)
{
    const char *str = "other_error";
    switch (ret_code)
    {
    case HttpRetCode::Success:
        str = "success";
        break;
    case HttpRetCode::InvalidRequest:
        str = "invalid_request";
        break;
    case HttpRetCode::ConnectionError:
        str = "connection_error";
        break;
    case HttpRetCode::SendTimeout:
        str = "send_timeout";
        break;
    case HttpRetCode::SendDisconnected:
        str = "send_disconnected";
        break;
    case HttpRetCode::SendError:
        str = "send_error";
        break;
    case HttpRetCode::RecvTimeout:
        str = "recv_timeout";
        break;
    case HttpRetCode::RecvDisconnected:
        str = "recv_disconnected";
        break;
    case HttpRetCode::RecvError:
        str = "recv_error";
        break;
    default:
        break;
    }

    return str;
}

HttpRetCode HttpClient::Execute(HttpResponse &response, const HttpRequest &request,
    int timeout, int connect_timeout, uint32_t max_body_length)
{
    HttpResponseCombo res_combo(&response, nullptr);
    HttpRetCode ret_code = ExecuteInner(res_combo, request,
        timeout, connect_timeout, max_body_length);
    return ret_code;
}

HttpRetCode HttpClient::Execute(HttpResponseReader &res_reader, const HttpRequest &request,
    int timeout, int connect_timeout, uint32_t max_body_length)
{
    res_reader.streaming_data_.strm = &res_reader.strm_;
    res_reader.streaming_data_.state = HttpMessageStream::UpdatingState::Header;
    res_reader.client_ = this;

    HttpResponseCombo res_combo(nullptr, &res_reader);
    HttpRetCode ret_code = ExecuteInner(res_combo, request,
        timeout, connect_timeout, max_body_length);
    return ret_code;
}

HttpRetCode HttpClient::ExecuteInner(HttpResponseCombo &res_combo, const HttpRequest &request,
    int timeout, int connect_timeout, uint32_t max_body_length)
{
    HttpResponse *response = res_combo.reader != nullptr ? &res_combo.reader->res_ : res_combo.res;
    if (response == nullptr) {
        return HttpRetCode::InvalidRequest;
    }

    HttpStreamingData *streaming_data = res_combo.reader == nullptr
        ? nullptr : &res_combo.reader->streaming_data_;

    HttpRetCode ret_code = HttpRetCode::Success;
    auto start_tm = chrono::steady_clock::now();

    if (connect_timeout == 0) {
        connect_timeout = min((int)50, timeout / 3);
    }

    UrlInfo url_info;
    url_info.Parse(request.url);
    if (url_info.addr.empty()) {
        return HttpRetCode::InvalidRequest;
    }

    bool is_https = String::CaseCmp(url_info.protocol_name, "https") == 0;
    bool use_proxy = !request.proxy_url.empty();
    string request_str;
    bool use_full_url = use_proxy;
    BuildRequestString(request_str, request, url_info, use_full_url);

    if (is_https && !use_proxy)
    {
#ifdef ENABLE_HTTPS
        BioStream bio_strm;
        bool is_succ = CreateHttpsConnection(bio_strm, url_info);
        if (!is_succ || bio_strm.ptr == nullptr) {
            return HttpRetCode::ConnectionError;
        }

        int bytes_to_write = (int)request_str.size();
        int bytes_written = BIO_write(bio_strm.ptr, request_str.c_str(), bytes_to_write);
        if (bytes_written != bytes_to_write) {
            return HttpRetCode::SendError;
        }

        //LogKeyInfo("Receiving Https response...");
        ret_code = ReceiveHttpsResponse(*response, bio_strm, max_body_length);
        //LogKeyInfo("Status code: %d", response.status_code);

        bio_strm.Clear();
        //LogKeyInfo("Before returning...");
        return ret_code;
#else
        LogError("HTTPS is not supported in the current settings. Please define ENABLE_HTTPS");
        return HttpRetCode::ConnectionError;
#endif //ENABLE_HTTPS
    }

    if (use_proxy) {
        url_info.Parse(request.proxy_url);
    }

    SocketInfo *socket_info = socket_cache_.Get(url_info.addr_and_port);
    if (socket_info == nullptr)
    {
        socket_info = new SocketInfo;
        socket_cache_.Add(url_info.addr_and_port, socket_info);
    }

    if (streaming_data != nullptr) {
        streaming_data->socket_info = socket_info;
    }

    int try_times = socket_info->is_connected ? 2 : 1;
    for (int try_idx = 0; try_idx < try_times; try_idx++)
    {
        //auto now_tm = steady_clock::now();
        //int elapsed_time = (int)chrono::duration_cast<chrono::milliseconds>(now_tm - socket_info->start_time).count();
        //if (socket_info->is_connected && elapsed_time >= 10 * 1000)
        //{
        //    //LogWarning("Close and reconnect (elapsed time of this connection: %d ms)", elapsed_time);
        //    socket_info->Close();
        //}

        if (!socket_info->is_connected)
        {
            if (try_idx > 0)
            {
                LogKeyInfo("Reconnecting to %s. Timeout: %d",
                    url_info.addr_and_port.c_str(), connect_timeout);
            }

            bool is_succ = socket_info->socket.Initialize();
            if (!is_succ)
            {
                LogError("Failed to initialize the client socket");
                return HttpRetCode::ConnectionError;
            }

            socket_info->is_connected = socket_info->socket.Connect(
                url_info.addr, url_info.port, connect_timeout);
            if (!socket_info->is_connected)
            {
                socket_info->socket.Close();
                LogWarning("Connecting error (timeout: %d)", connect_timeout);
                return HttpRetCode::ConnectionError;
            }

            socket_info->start_time = steady_clock::now();
        }

        socket_info->socket.SetSendTimeout(timeout);

        if (use_proxy)
        {
            stringstream request_stream;
            request_stream << "CONNECT " << url_info.addr_and_port << " HTTP/1.1\r\n"
                << "User-Agent: " << request.user_agent << "\r\n\r\n";
            string proxy_request_str = request_stream.str();

            HttpResponse proxy_response;
            ret_code = SendAndReceive(*response, streaming_data, proxy_request_str,
                *socket_info, start_tm, timeout, true, max_body_length);
            if (ret_code != HttpRetCode::Success || response->status_code != 200)
            {
                ret_code = HttpRetCode::ProxyError;
            }
        }

        if (ret_code == HttpRetCode::Success)
        {
            ret_code = SendAndReceive(*response, streaming_data, request_str,
                *socket_info, start_tm, timeout, false, max_body_length);
            if (response->status_code == 302 || response->status_code == 303
                || response->status_code == 307) //redirection
            {
                string server_str = GetRedirectionServer(*response);
                url_info.Parse(server_str);
                if (String::CaseCmp(url_info.protocol_name, "http") == 0)
                {
                    BuildRequestString(request_str, request, url_info, use_full_url);
                    ret_code = SendAndReceive(*response, streaming_data, request_str,
                        *socket_info, start_tm, timeout, false, max_body_length);
                }
            }
        }

        //it is very important to close the socket upon failure,
        //to avoid negative impact on the next Execute() call
        if (ret_code != HttpRetCode::Success
            || (streaming_data == nullptr && !response->keep_alive))
        {
            socket_info->Close();
        }

        //retry only upon ever-disconnection (detected by a sending or receiving error)
        if (ret_code != HttpRetCode::SendDisconnected
            && ret_code != HttpRetCode::RecvDisconnected)
        {
            break;
        }

        if (ret_code != HttpRetCode::Success)
        {
            LogWarning("%d of %d: ret_code: %u (%s)", try_idx, try_times,
                ret_code, RetCodeToString(ret_code));
        }
    }

    return ret_code;
}

HttpRetCode HttpClient::SendAndReceive(HttpResponse &response,
    HttpStreamingData *streaming_data,
    const string &request_str, SocketInfo &socket_info,
    const time_point<steady_clock> &start_tm,
    int timeout, bool is_proxy, uint32_t max_body_length) const
{
    bool is_end_upon_empty_line = is_proxy;
    HttpRetCode ret_code = HttpRetCode::Success;
    Socket::RetCode sock_ret = socket_info.socket.WriteAll(
        request_str.c_str(), (int)request_str.size());
    switch (sock_ret)
    {
    case Socket::RetCode::Success:
        ret_code = HttpRetCode::Success;
        break;
    case Socket::RetCode::Disconnected:
        ret_code = HttpRetCode::SendDisconnected;
        break;
    default:
        ret_code = HttpRetCode::SendError;
        break;
    }

    if (ret_code == HttpRetCode::Success)
    {
        auto now_tm = chrono::steady_clock::now();
        int elapsed_time = (int)chrono::duration_cast<chrono::milliseconds>(now_tm - start_tm).count();
        if (timeout > elapsed_time)
        {
            ret_code = ReceiveResponse(response, streaming_data, socket_info,
                timeout - elapsed_time, is_end_upon_empty_line, max_body_length);
        }
        else
        {
            ret_code = HttpRetCode::SendTimeout;
        }
    }

    return ret_code;
}

HttpRetCode HttpClient::ReceiveResponse(HttpResponse &response,
    HttpStreamingData *streaming_data, SocketInfo &socket_info,
    int timeout, bool is_end_upon_empty_line, uint32_t max_body_length,
    bool clear_response_first) const
{
    HttpRetCode ret_code = HttpRetCode::Success;
    if (clear_response_first) {
        response.Clear();
    }
    Socket &socket = socket_info.socket;

    socket.SetRecvTimeout(min(10, max(1, timeout / 5)));

    auto start_tm = chrono::steady_clock::now();
    std::chrono::steady_clock::time_point cur_tm;

    Socket::RetCode sock_ret = Socket::RetCode::Success;
    HttpMessageStream::HeaderInfo hdr;
    HttpMessageStream::UpdatingState state = HttpMessageStream::UpdatingState::Header;
    if (streaming_data != nullptr)
    {
        state = streaming_data->state;
        hdr = streaming_data->hdr;
    }

    HttpMessageStream strm;
    HttpMessageStream *stream_ptr = streaming_data != nullptr ? streaming_data->strm : &strm;
    int bytes_read = 0;
    bool be_continue = true;
    while (be_continue)
    {
        sock_ret = socket.RecvEx(bytes_read, buf_, buf_len_);
        if (sock_ret == Socket::RetCode::Success)
        {
            if ((uint32_t)(stream_ptr->str.size() + bytes_read) > max_body_length)
            {
                //LogWarning("Warning: Data size exceeds %u (max_body_length)", max_body_length);
                ret_code = HttpRetCode::RecvError;
                break;
            }

            //LogKeyInfo("bytes_read: %d", bytes_read);
            //if (bytes_read == 1) {
            //    LogKeyInfo("byte: %c", buf_[0]);
            //}

            stream_ptr->str.append(buf_, bytes_read);

            int chunk_n1 = streaming_data != nullptr ? (int)streaming_data->chunks.size() : 0;
            stream_ptr->UpdateMessage(response, hdr, state, streaming_data, header_field_map_);

            if (streaming_data != nullptr)
            {
                streaming_data->state = state;
                streaming_data->hdr = hdr;
            }

            if (state == HttpMessageStream::UpdatingState::End) {
                break;
            }

            if (is_end_upon_empty_line)
            {
                uint32_t len = (uint32_t)stream_ptr->str.size();
                if (stream_ptr->str == "\r\n" || (len > 4 && stream_ptr->str.substr(len - 4) == "\r\n\r\n"))
                {
                    state = HttpMessageStream::UpdatingState::End;
                    break;
                }
            }

            if (streaming_data != nullptr)
            {
                int chunk_n2 = (int)streaming_data->chunks.size();
                if (state == HttpMessageStream::UpdatingState::Body
                    && chunk_n2 > chunk_n1)
                {
                    break;
                }
            }
        }
        else if (sock_ret != Socket::RetCode::Timeout)
        {
            break;
        }

        /*int data_size = socket.DataSizeAvailableForRead();
        if (data_size > 0)
        {
            sock_ret = socket.ReadAll(buf, min(buf_len, data_size));
            if (sock_ret != Socket::RetCode::Success) {
                break;
            }

            strm.str.append(buf, data_size);
        }
        else if(!strm.str.empty())
        {
            strm.UpdateMessage(response, hdr, state, header_field_map_);
            if (state == HttpMessageStream::UpdatingState::End) {
                break;
            }
        }*/

        cur_tm = chrono::steady_clock::now();
        int elapsed_time = (int)chrono::duration_cast<chrono::milliseconds>(cur_tm - start_tm).count();
        if (elapsed_time >= timeout)
        {
            sock_ret = Socket::RetCode::Timeout;
            break;
        }

        Thread::SleepMicro(100);
    }

    if (!response.header_lines.empty())
    {
        const auto &status_line = response.header_lines[0];
        size_t pos = status_line.find(' ');
        if (pos != string::npos)
        {
            string code_str = status_line.substr(pos);
            String::TrimLeft(code_str);
            response.status_code = String::ToInt32(code_str);
        }
    }

    if (ret_code == HttpRetCode::Success)
    {
        switch (sock_ret)
        {
        case Socket::RetCode::Success:
            if (streaming_data == nullptr)
            {
                ret_code = state == HttpMessageStream::UpdatingState::End
                    ? HttpRetCode::Success : HttpRetCode::RecvError;
            }
            else
            {
                ret_code = state >= HttpMessageStream::UpdatingState::Body
                    ? HttpRetCode::Success : HttpRetCode::RecvError;
            }

            //if (ret_code != HttpRetCode::Success) {
            //    LogWarning("Warning: stream updating state = %u", state);
            //}

            break;
        case Socket::RetCode::Disconnected:
            ret_code = HttpRetCode::RecvDisconnected;
            break;
        case Socket::RetCode::Timeout:
            ret_code = HttpRetCode::RecvTimeout;
            break;
        default:
            //LogWarning("Warning: sock_ret = %u", sock_ret);
            ret_code = HttpRetCode::RecvError;
            break;
        }
    }

    response.keep_alive = hdr.keep_alive && ret_code == HttpRetCode::Success;
    return ret_code;
}

//static
void HttpClient::UrlInfo::Parse(const string &url_str)
{
    string::size_type offset = url_str.find("://");
    size_t pos0 = offset == string::npos ? 0 : offset;
    size_t pos1 = offset == string::npos ? 0 : offset + 3;
    size_t pos2 = url_str.find(":", pos1);
    size_t pos3 = url_str.find("/", pos1);
    if (pos3 == string::npos) {
        pos3 = url_str.size();
    }
    if (pos2 == string::npos) {
        pos2 = pos3;
    }

    auto &url_info = *this;
    url_info.url = url_str;
    url_info.protocol_name = url_str.substr(0, pos0);
    url_info.addr = url_str.substr(pos1, pos2 - pos1);
    url_info.port = String::CaseCmp(url_info.protocol_name, "https") == 0 ? 443 : 80;
    if (pos3 > pos2) {
        url_info.port = String::ToInt32(url_str.substr(pos2 + 1, pos3 - pos2 - 1));
    }
    url_info.addr_and_port = url_info.addr + ":" + to_string(url_info.port);
    url_info.resource_part = url_str.substr(pos3);
    if (url_info.resource_part.empty()) {
        url_info.resource_part = "/";
    }
}

//static
void HttpClient::BuildRequestString(string &request_str,
    const HttpRequest &request, const UrlInfo &url_info,
    bool use_full_url)
{
    stringstream request_stream;
    const char *accept_type = "*/*";
    switch (request.type)
    {
    case HttpRequest::Method::Post:
        request_stream << "POST ";
        break;
    case HttpRequest::Method::Get:
    default:
        request_stream << "GET ";
        break;
    }

    string content_type_line;
    string content_type_field = "Content-Type:";
    int field_len = (int)content_type_field.size();
    for (const auto &hdr_line : request.header_lines)
    {
        if ((int)hdr_line.size() >= field_len
            && strncasecmp(content_type_field.c_str(), hdr_line.c_str(), field_len) == 0)
        {
            content_type_line = hdr_line;
            break;
        }
    }

    if (use_full_url)
    {
        request_stream << url_info.url << " HTTP/1.1\r\n";
    }
    else
    {
        request_stream << url_info.resource_part << " HTTP/1.1\r\n";
    }
    request_stream << "Host: " << url_info.addr;
    if (url_info.port != 80) {
        request_stream << ":" << url_info.port;
    }
    request_stream << "\r\n";
    if (content_type_line.empty()) {
        request_stream << "Content-Type: text/plain;charset=UTF-8" << "\r\n";
    }
    else {
        request_stream << content_type_line << "\r\n";
    }
    request_stream << "Accept: " << accept_type << "\r\n"
        << "Content-Length: " << request.body.size() << "\r\n";
    for (const auto &hdr_line : request.header_lines)
    {
        if (hdr_line.find("\n") == string::npos) {
            request_stream << hdr_line << "\r\n";
        }
    }
    request_stream << "\r\n" << request.body;
    request_str = request_stream.str();
}

string HttpClient::GetRedirectionServer(const HttpResponse & response)
{
    string server_str;
    for (const auto &header_line : response.header_lines)
    {
        char ch = header_line.empty() ? ' ' : header_line[0];
        if (ch != 'L' && ch != 'l') {
            continue;
        }

        auto pos = header_line.find(':');
        if (pos == string::npos) {
            continue;
        }

        if (String::CaseCmp(header_line.substr(0, pos), "Location") == 0)
        {
            server_str = header_line.substr(pos + 1);
            String::Trim(server_str);
            break;
        }
    }

    return server_str;
}

#ifdef ENABLE_HTTPS
bool HttpClient::CreateHttpsConnection(BioStream&strm, const UrlInfo &url_info) const
{
    strm.Clear();
    strm.ptr = BIO_new_ssl_connect(ssl_ctx_);
    if (strm.ptr == nullptr) {
        return false;
    }

    bool ret = SetupHttpsConnection(strm, url_info);
    if (!ret)
    {
        strm.Clear();
        return false;
    }

    return strm.ptr != nullptr;
}
#endif //ENABLE_HTTPS

#ifdef ENABLE_HTTPS
bool HttpClient::SetupHttpsConnection(BioStream&strm, const UrlInfo &url_info) const
{
    int res = BIO_set_conn_hostname(strm.ptr, url_info.addr_and_port.c_str());
    if (1 != res) {
        return false;
    }

    SSL *ssl = nullptr;
    BIO_get_ssl(strm.ptr, &ssl);
    if (ssl == nullptr) {
        return false;
    }

    const char *preferred_ciphers = "HIGH:!aNULL:!kRSA:!PSK:!SRP:!MD5:!RC4";
    res = SSL_set_cipher_list(ssl, preferred_ciphers);
    if (1 != res) {
        return false;
    }

    res = SSL_set_tlsext_host_name(ssl, url_info.addr.c_str());
    if (1 != res) {
        return false;
    }

    res = BIO_do_connect(strm.ptr);
    if (1 != res) {
        return false;
    }

    res = BIO_do_handshake(strm.ptr);
    if (1 != res) {
        return false;
    }

    //Step 1: verify a server certificate was presented during the negotiation
    X509 *cert = SSL_get_peer_certificate(ssl);
    if (nullptr == cert) {
        return false;
    }

    X509_free(cert); //Free immediately

    //Step 2: verify the result of chain verification
    //Verification performed according to RFC 4158
    res = SSL_get_verify_result(ssl);
    if (X509_V_OK != res) {
        return false;
    }

    //Step 3: hostname verification
    //todo

    return true;
}
#endif //ENABLE_HTTPS

#ifdef ENABLE_HTTPS
HttpRetCode HttpClient::ReceiveHttpsResponse(HttpResponse &response,
    BioStream &bio_strm, uint32_t max_body_length)
{
    HttpRetCode ret_code = HttpRetCode::Success;
    response.Clear();

    HttpMessageStream::HeaderInfo hdr;
    HttpMessageStream::UpdatingState state = HttpMessageStream::UpdatingState::Header;
    HttpMessageStream strm;
    int bytes_read = 0;
    do
    {
        bytes_read = BIO_read(bio_strm.ptr, buf_, buf_len_);
        //LogKeyInfo("We get %d bytes", bytes_read);
        if (bytes_read > 0)
        {
            if ((uint32_t)(strm.str.size() + bytes_read) > max_body_length)
            {
                ret_code = HttpRetCode::RecvError;
                break;
            }

            strm.str.append(buf_, bytes_read);
            strm.UpdateMessage(response, hdr, state, nullptr, header_field_map_);
            //LogKeyInfo("End of update (state = %d)", state);
            if (state == HttpMessageStream::UpdatingState::End) {
                break;
            }
        }
    } while (bytes_read > 0 || BIO_should_retry(bio_strm.ptr));

    //LogKeyInfo("Extracting status code...");
    if (!response.header_lines.empty())
    {
        const auto &status_line = response.header_lines[0];
        size_t pos = status_line.find(' ');
        if (pos != string::npos)
        {
            string code_str = status_line.substr(pos);
            String::TrimLeft(code_str);
            response.status_code = String::ToInt32(code_str);
        }
    }

    return ret_code;
}
#endif //ENABLE_HTTPS

#ifdef ENABLE_HTTPS
bool HttpClient::SetCertificate(const char *buf, int length)
{
    BioStream bio_strm;
    bio_strm.ptr = BIO_new_mem_buf((void*)buf, length);
    if (bio_strm.ptr == nullptr) {
        return false;
    }

    X509_STORE *cts = SSL_CTX_get_cert_store(ssl_ctx_);
    if (cts == nullptr) {
        return false;
    }

    STACK_OF(X509_INFO) *info = PEM_X509_INFO_read_bio(bio_strm.ptr,
        nullptr, nullptr, nullptr);
    if (info == nullptr) {
        return false;
    }

    //iterate over all entries and add them to the x509_store one by one
    int count = sk_X509_INFO_num(info);
    for (int idx = 0; idx < count; idx++)
    {
        X509_INFO *info_value = sk_X509_INFO_value(info, idx);
        if (info_value->x509) {
            X509_STORE_add_cert(cts, info_value->x509);
        }
        if (info_value->crl) {
            X509_STORE_add_crl(cts, info_value->crl);
        }
    }

    sk_X509_INFO_pop_free(info, X509_INFO_free); //cleanup
    return true;
}
#endif //ENABLE_HTTPS

} //end of namespace
