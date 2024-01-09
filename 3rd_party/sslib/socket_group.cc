#include "socket_group.h"

using namespace std;

namespace sslib
{

SocketGroup::SocketGroup(bool need_lock)
{
    need_lock_ = need_lock;
}

SocketGroup::~SocketGroup()
{
    if (need_lock_) {
        lock_.lock();
    }

    for (int i = 0; i < (int)socket_list_.size(); i++)
    {
        Socket *pSock = socket_list_[i];
        delete pSock;
    }
    socket_list_.clear();

    if (need_lock_) {
        lock_.unlock();
    }
}

void SocketGroup::Add(Socket *socket)
{
    if (need_lock_) {
        lock_.lock();
    }
	socket_list_.push_back(socket);
    if (need_lock_) {
        lock_.unlock();
    }
}

void SocketGroup::Remove(Socket *socket)
{
    if (need_lock_) {
        lock_.lock();
    }

	vector<Socket*>::iterator iter;
	for(iter = socket_list_.begin(); iter != socket_list_.end(); iter++)
	{
		Socket *ptr = *iter;
		if(ptr == socket)
		{
			delete ptr;
			socket_list_.erase(iter);
			break;
		}
	}

    if (need_lock_) {
        lock_.unlock();
    }
}

void SocketGroup::Remove(int idx)
{
    if (need_lock_) {
        lock_.lock();
    }

	vector<Socket*>::iterator iter = socket_list_.begin() + idx;
	Socket *socket = *iter;
	delete socket;
	socket_list_.erase(iter);

    if (need_lock_) {
        lock_.unlock();
    }
}

} //end of namespace
