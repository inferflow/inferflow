#pragma once

#include <vector>
#include "macro.h"
#include "binary_file_stream.h"
#include "log.h"

namespace sslib
{

template <class EleType>
class PtrVector : public std::vector<EleType*>
{
public:
    PtrVector() {};
    virtual ~PtrVector()
    {
        Clear(true);
    }

    void clear() = delete;

    void Clear(bool is_deep = true)
    {
        if(is_deep)
        {
            for(size_t idx = 0; idx < std::vector<EleType*>::size(); idx++)
            {
                if((*this)[idx] != nullptr)
                {
                    delete (*this)[idx];
                    (*this)[idx] = nullptr;
                }
            }
        }
        std::vector<EleType*>::clear();
    }
};

class VectorHelper
{
public:
    template <class ElementType>
    static bool Write(const std::vector<ElementType> &v, const std::string &file_path)
    {
        BinaryFileStream strm;
        bool ret = strm.OpenForWrite(file_path);
        Macro_RetxFalseIf(!ret, LogError("Failed to open file %s", file_path.c_str()));

        ret = Write(v, strm);
        ret = ret && strm.Flush();
        return ret && strm.IsGood();
    }

    template <class ElementType>
    static bool Read(std::vector<ElementType> &v, const std::string &file_path)
    {
        BinaryFileStream strm;
        bool ret = strm.OpenForRead(file_path);
        Macro_RetxFalseIf(!ret, LogError("Failed to open file %s", file_path.c_str()));

        ret = Read(v, strm);
        return ret && strm.IsGood();
    }

    template <class ElementType>
    static bool Write(const std::vector<ElementType> &v, IBinaryStream &strm)
    {
        uint32_t item_num = (uint32_t)v.size();
        bool ret = strm.Write(item_num);
        for (uint32_t iItem = 0; ret && iItem < item_num; ++iItem)
        {
            const auto &theElement = v[iItem];
            ret = theElement.Write(strm);
        }

        return ret && strm.IsGood();
    }

    template <class ElementType>
    static bool Read(std::vector<ElementType> &v, IBinaryStream &strm)
    {
        v.clear();
        uint32_t item_num = 0;
        bool ret = strm.Read(item_num);

        v.resize(item_num);
        for (uint32_t iItem = 0; ret && iItem < item_num; iItem++)
        {
            ret = v[iItem].Read(strm);
        }

        ret = ret && strm.IsGood();
        return ret;
    }

    template <class ElementType>
    static bool Print(const std::vector<ElementType> &v, const std::string &file_path)
    {
        std::ofstream strm(file_path);
        Macro_RetxFalseIf(!strm, LogError("Failed to open file %s", file_path.c_str()));

        bool ret = Print(v, strm);
        strm.close();
        return ret && strm.good();
    }

    template <class ElementType>
    static bool Print(const std::vector<ElementType> &v, ostream &strm)
    {
        bool ret = true;
        uint32_t item_num = (uint32_t)v.size();
        strm << "; " << item_num << endl;
        for (uint32_t iItem = 0; ret && iItem < item_num; ++iItem)
        {
            const auto &theElement = v[iItem];
            ret = theElement.Print(strm);
            strm << endl;
        }

        return ret && strm.good();
    }
};

} //end of namespace
