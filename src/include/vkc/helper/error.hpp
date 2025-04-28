#pragma once

#include <source_location>
#include <string>

namespace vkc {

class Error {
public:
    int code;
    std::source_location source;
    std::string msg;

    explicit Error(int code, const std::source_location& source = std::source_location::current());
    Error(int code, const std::string& msg, const std::source_location& source = std::source_location::current());
    Error(int code, std::string&& msg, const std::source_location& source = std::source_location::current());
    Error(const Error& rhs) = default;
    Error(Error&& rhs) noexcept = default;
};

}  // namespace vkc

#ifdef _TLCT_LIB_HEADER_ONLY
#    include "vkc/helper/error.cpp"
#endif
