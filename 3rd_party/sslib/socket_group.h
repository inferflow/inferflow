#pragma once

#include <vector>
#include <mutex>
#include "socket.h"
#include "thread.h"

namespace sslib
{

class SocketGroup
{
public:
	SocketGroup(bool need_lock = true);
	virtual ~SocketGroup();

	void Add(Socket *socket);
	void Remove(Socket *socket);
	void Remove(int idx);

protected:
	std::vector<Socket*> socket_list_;
	std::mutex lock_;

	//whether the thread group is thread-safe
	bool need_lock_ = true;
};

} //end of namespace
