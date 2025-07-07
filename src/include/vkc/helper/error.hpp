#pragma once

#include <source_location>
#include <string>

#include "vkc/helper/vulkan.hpp"

namespace vkc {

enum class ECode {
    eUnknown = 0,
    eUnexValue = 1,        // Unexpected Value
    eNoSupport = 2,        // Feature Not Supported
    eNoImpl = 3,           // Not Implemented
    eResourceInvalid = 3,  // Resource is Invalid
};

enum class ECate {
    eSuccess = 0,
    eUnknown,
    eMisc,
    eVkC,
    eVk,
    eStb,
    eGLFW,
};

constexpr std::string_view errCateToStr(const ECate cate) noexcept {
    switch (cate) {
        case ECate::eSuccess:
            return "Success";
        case ECate::eUnknown:
            return "Unknown";
        case ECate::eMisc:
            return "Misc";
        case ECate::eVkC:
            return "VkC";
        case ECate::eVk:
            return "Vulkan";
        case ECate::eStb:
            return "Stb";
        case ECate::eGLFW:
            return "GLFW";
        default:
            return "Unknown";
    }
}

class Error {
public:
    ECate cate;
    int code;
    std::source_location source;
    std::string msg;

    Error() noexcept;

    template <typename T>
    Error(ECate cate, T code, const std::source_location& source = std::source_location::current()) noexcept;

    template <typename T>
    Error(ECate cate, T code, std::string&& msg,
          const std::source_location& source = std::source_location::current()) noexcept;
    ;
    Error& operator=(const Error& rhs) = default;
    Error(const Error& rhs) = default;
    Error& operator=(Error&& rhs) = default;
    Error(Error&& rhs) noexcept = default;
};

template <typename T>
Error::Error(const ECate cate, const T code, const std::source_location& source) noexcept
    : cate(cate), code((int)code), source(source) {}

template <typename T>
Error::Error(const ECate cate, const T code, std::string&& msg, const std::source_location& source) noexcept
    : cate(cate), code((int)code), source(source), msg(std::move(msg)) {}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/error.cpp"
#endif
