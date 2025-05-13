#include <source_location>
#include <string>
#include <utility>

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/error.hpp"
#endif

namespace vkc {

Error::Error(const vk::Result code, const std::source_location& source) : code((int)code), source(source) {}

Error::Error(const int code, const std::source_location& source) : code(code), source(source) {}

Error::Error(const int code, const std::string& msg, const std::source_location& source)
    : code(code), source(source), msg(msg) {}

Error::Error(const int code, std::string&& msg, const std::source_location& source)
    : code(code), source(source), msg(std::move(msg)) {}

}  // namespace vkc
