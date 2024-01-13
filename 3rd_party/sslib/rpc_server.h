#include "base_tcp_server.h"
#include "vector_ex.h"

namespace sslib
{

class BaseRpcServer : public BaseTcpServer
{
public:
    BaseRpcServer();
    virtual ~BaseRpcServer();

    void Clear();

    bool Init(int port, int worker_count, bool is_study_mode = false);
    bool Start();
    void Stop();

protected:
    static const uint32_t MagicCode = ((31415 << 16) | 27182);

protected:
    virtual bool HandleRequest(string &response, const string &request);

    virtual bool PrepareTask(ConnectionData *connection,
        const char *data, uint32_t data_len) override;
    virtual bool HandleTask(TcpTask *task) override;

private:
    BaseRpcServer(const BaseRpcServer &rhs) = delete;
    BaseRpcServer& operator = (const BaseRpcServer &rhs) = delete;
};

} //end of namespace
