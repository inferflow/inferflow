#pragma once

#include "socket.h"

namespace sslib
{

class ISocketSerializable
{
public:
	virtual bool Read(Socket *sock) = 0;
	virtual bool Write(Socket *sock) const = 0;

	virtual ~ISocketSerializable() {};
};

} //end of namespace
