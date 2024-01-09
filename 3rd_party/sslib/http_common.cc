#include "http_common.h"
#include "log.h"

namespace sslib
{

using namespace std;

HttpMessage::~HttpMessage()
{
    Clear();
}

void HttpMessage::Clear()
{
    header_lines.clear();
    body_length = 0;
    body.clear();
}

void HttpMessage::AddConnectionHeader(bool keep_alive)
{
    const char *line_str = keep_alive ? "Connection: keep-alive" : "Connection: close";
    header_lines.push_back(line_str);
}

void HttpMessageStream::UpdateMessage(HttpMessage &msg, HeaderInfo &hdr,
    UpdatingState &state, HttpStreamingData *streaming_data,
    const HttpHeaderFieldMap &header_field_map)
{
    bool is_close = false;
    string line_str, field_name, field_value;
    while (state != UpdatingState::End)
    {
        if (state == UpdatingState::Body)
        {
            if (!hdr.is_chunked_transfer_encoding)
            {
                //if (is_close && msg.body_length == 0) {
                if (msg.body_length == 0) {
                    msg.body_length = (int)(this->str.length() - this->offset);
                }

                //LogKeyInfo("std encoding, %d, %d, %d", this->offset, msg.body_length, this->str.size());
                if (msg.body_length > 0 && this->offset + msg.body_length <= this->str.size())
                {
                    msg.body.append(this->str.c_str() + this->offset, msg.body_length);
                    this->offset += msg.body_length;
                    state = UpdatingState::End;
                    //LogKeyInfo("UpdatingState::End r1");
                }
                return;
            }
            else //chunked_transfer_encoding
            {
                if (hdr.chunk_length <= 0)
                {
                    //LogKeyInfo("chunked encoding");
                    this->ReadCompleteLine(line_str);
                    //LogKeyInfo("line_str: %s", line_str.c_str());
                    if (line_str.empty()) {
                        return;
                    }

                    hdr.chunk_length = GetChunkLength(line_str);
                }

                if (hdr.chunk_length == 0)
                {
                    state = UpdatingState::End;
                    //LogKeyInfo("UpdatingState::End r2");
                    break;
                }

                if (this->offset + hdr.chunk_length + 2 <= this->str.size())
                {
                    if (streaming_data == nullptr)
                    {
                        msg.body.append(this->str.c_str() + this->offset, hdr.chunk_length);
                    }
                    else
                    {
                        string chunk_str(this->str.c_str() + this->offset, hdr.chunk_length);
                        //LogKeyInfo("chunk_str: %s", chunk_str.c_str());
                        streaming_data->chunks.push_back(chunk_str);
                    }

                    this->offset += hdr.chunk_length;

                    if (this->offset + 2 <= this->str.size()
                        && this->str[this->offset] == '\r'
                        && this->str[this->offset + 1] == '\n')
                    {
                        this->offset += 2;
                    }

                    hdr.chunk_length = 0;
                    continue;
                }
                else
                {
                    return;
                }
            }
        }

        this->ReadCompleteLine(line_str);
        if (line_str.empty())
        {
            if (this->offset < str.length())
            {
                state = UpdatingState::Body;
                continue;
            }
            break;
        }

        if (msg.header_lines.empty())
        {
            msg.header_lines.push_back(line_str);
        }
        else
        {
            String::Trim(line_str);
            if (line_str.empty())
            {
                state = UpdatingState::Body;
                continue;
            }

            msg.header_lines.push_back(line_str);
            size_t pos = line_str.find(':');
            if (pos == string::npos) {
                continue;
            }

            field_name = line_str.substr(0, pos);
            field_value = line_str.substr(pos + 1);
            String::Trim(field_value);

            auto iter = header_field_map.find(field_name);
            if (iter == header_field_map.end()) {
                continue;
            }

            switch (iter->second)
            {
            case HttpHeaderFieldId::Connection:
                is_close = strcasecmp(field_value.c_str(), "close") == 0;
                if (is_close) {
                    hdr.is_close = true;
                }
                if (strcasecmp(field_value.c_str(), "keep-alive") == 0) {
                    hdr.keep_alive = true;
                }
                break;
            case HttpHeaderFieldId::TransferEncoding:
                hdr.is_chunked_transfer_encoding = strcasecmp(field_value.c_str(), "chunked") == 0;
                break;
            case HttpHeaderFieldId::ContentLength:
                msg.body_length = String::ToInt32(field_value);
                break;
            default:
                break;
            }
        }
    }
}

bool HttpMessageStream::ReadCompleteLine(string &line_str)
{
    line_str.clear();
    size_t idx = offset;
    size_t line_end = offset;
    while (idx + 2 <= str.size())
    {
        if (str[idx] == '\r' && str[idx + 1] == '\n')
        {
            line_end = idx + 2;
            break;
        }

        idx++;
    }

    if (line_end > offset)
    {
        line_str = str.substr(offset, line_end - offset);
        offset = line_end;
    }

    return !line_str.empty();
}

//static
int HttpMessageStream::GetChunkLength(const string &line_str)
{
    int chunk_length = 0;
    for (size_t char_idx = 0; char_idx < line_str.size(); char_idx++)
    {
        char ch = line_str[char_idx];
        if (ch >= '0' && ch <= '9') {
            chunk_length = 16 * chunk_length + (ch - '0');
        }
        else if (ch >= 'A' && ch <= 'F') {
            chunk_length = 16 * chunk_length + (ch - 'A' + 10);
        }
        else if (ch >= 'a' && ch <= 'f') {
            chunk_length = 16 * chunk_length + (ch - 'a' + 10);
        }
        else {
            break;
        }
    }

    return chunk_length;
}

} //end of namespace
