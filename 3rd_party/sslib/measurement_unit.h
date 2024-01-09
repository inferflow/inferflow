#pragma once

namespace sslib
{

enum class InformationUnit
{
    Unknown = 0,
    Bit = 1,
    Byte,
    Kilobyte, KB = Kilobyte,
    Megabyte, MB = Megabyte,
    Gigabyte, GB = Gigabyte,
    Terabyte, TB = Terabyte,
    Petabyte, PB = Petabyte,
    Exabyte, EB = Exabyte,
    Zettabyte, ZB = Zettabyte,
    Yottabyte, YB = Yottabyte
};

} //end of namespace
