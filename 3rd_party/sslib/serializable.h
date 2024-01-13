#pragma once

#include "prime_types.h"
#include "binary_stream.h"
#include <iostream>

namespace sslib
{

class ISerializable
{
public:
	virtual bool Read(std::istream *reader)
	{ 
        (void)reader;
		return false; 
	};
	virtual bool Write(std::ostream *writer) const
	{ 
        (void)writer;
		return false; 
	};

	virtual bool Read(std::istream &reader, void *params = nullptr)
	{
        (void)reader; (void)params;
		return false; 
	};
	virtual bool Write(std::ostream &writer, void *params = nullptr) const
	{
        (void)writer; (void)params;
		return false; 
	};

	virtual ~ISerializable(){};
};

class IBinSerializable
{
public:
    virtual bool Read(IBinStream &strm, void *params = nullptr) = 0;
    virtual bool Write(IBinStream &strm, void *params = nullptr) const = 0;

    virtual bool Print(std::ostream &ps, const std::string &options = "") const
    {
        (void)options; return ps.good();
    }

    virtual ~IBinSerializable(){};
};

} //end of namespace
