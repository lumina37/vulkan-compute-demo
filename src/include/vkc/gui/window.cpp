#include <expected>
#include <memory>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/window.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

WindowBox::WindowBox(vk::Extent2D extent, GLFWwindow* window) noexcept : extent_(extent), window_(window) {}

WindowBox::WindowBox(WindowBox&& rhs) noexcept : extent_(rhs.extent_), window_(std::exchange(rhs.window_, nullptr)) {}

WindowBox::~WindowBox() noexcept {
    if (window_ == nullptr) return;
    glfwDestroyWindow(window_);
    window_ = nullptr;
}

std::expected<void, Error> WindowBox::globalInit() noexcept {
    const int initRes = glfwInit();
    if (initRes == GLFW_FALSE) {
        return std::unexpected{Error{ECate::eGLFW, 0, "failed to init GLFW"}};
    }
    return {};
}

void WindowBox::globalDestroy() noexcept { glfwTerminate(); }

std::expected<std::vector<std::string_view>, Error> WindowBox::getExtensions() {
    uint32_t count;
    const auto pExts = glfwGetRequiredInstanceExtensions(&count);
    if (pExts == nullptr) {
        return std::unexpected{Error{ECate::eGLFW, 0, "failed to get GLFW extensions"}};
    }

    auto exts = rgs::views::iota(0, (int)count) |
                rgs::views::transform([pExts](const int i) { return std::string_view{pExts[i]}; }) |
                rgs::to<std::vector>();

    return exts;
}

std::expected<WindowBox, Error> WindowBox::create(const vk::Extent2D extent) noexcept {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    auto window = glfwCreateWindow((int)extent.width, (int)extent.height, "Vulkan Graphics Demo", nullptr, nullptr);
    if (window == nullptr) {
        return std::unexpected{Error{ECate::eGLFW, 0, "failed to create GLFW window"}};
    }

    return WindowBox{extent, window};
}

}  // namespace vkc
