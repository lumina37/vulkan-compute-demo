#pragma once

#include <source_location>
#include <string>

#include "vkc/helper/vulkan.hpp"

namespace vkc {

class Error {
public:
    int code;
    std::source_location source;
    std::string msg;

    Error() = default;
    explicit Error(vk::Result code, const std::source_location& source = std::source_location::current());
    explicit Error(int code, const std::source_location& source = std::source_location::current());
    Error(int code, const std::string& msg, const std::source_location& source = std::source_location::current());
    Error(int code, std::string&& msg, const std::source_location& source = std::source_location::current());
    Error(const Error&) = default;
    Error(Error&&) noexcept = default;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/error.cpp"
#endif
