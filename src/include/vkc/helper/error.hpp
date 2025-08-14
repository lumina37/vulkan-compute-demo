#pragma once

#include <source_location>
#include <string>

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

    Error() noexcept : cate(ECate::eSuccess), code(0) {}

    template <typename T>
    Error(ECate cate, T code, const std::source_location& source = std::source_location::current()) noexcept
        : cate(cate), code((int)code), source(source) {}

    template <typename T>
    Error(ECate cate, T code, std::string&& msg,
          const std::source_location& source = std::source_location::current()) noexcept
        : cate(cate), code((int)code), source(source), msg(std::move(msg)) {}

    Error& operator=(const Error& rhs) = default;
    Error(const Error& rhs) = default;
    Error& operator=(Error&& rhs) = default;
    Error(Error&& rhs) noexcept = default;
};

}  // namespace vkc
