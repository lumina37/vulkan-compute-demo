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

WindowManager::WindowManager(vk::Extent2D extent, GLFWwindow* window) noexcept : extent_(extent), window_(window) {}

WindowManager::WindowManager(WindowManager&& rhs) noexcept
    : extent_(rhs.extent_), window_(std::exchange(rhs.window_, nullptr)) {}

WindowManager::~WindowManager() noexcept {
    if (window_ == nullptr) return;
    glfwDestroyWindow(window_);
    window_ = nullptr;
}

std::expected<void, Error> WindowManager::globalInit() noexcept {
    const int initRes = glfwInit();
    if (initRes == GLFW_FALSE) {
        return std::unexpected{Error{-1, "failed to init GLFW"}};
    }
    return {};
}

void WindowManager::globalDestroy() noexcept { glfwTerminate(); }

std::expected<std::vector<std::string_view>, Error> WindowManager::getExtensions() {
    uint32_t count;
    const auto pExts = glfwGetRequiredInstanceExtensions(&count);
    if (pExts == nullptr) {
        return std::unexpected{Error{-1, "failed to get GLFW extensions"}};
    }

    auto exts = rgs::views::iota(0, (int)count) |
                rgs::views::transform([pExts](const int i) { return std::string_view{pExts[i]}; }) |
                rgs::to<std::vector>();

    return exts;
}

std::expected<WindowManager, Error> WindowManager::create(const vk::Extent2D extent) noexcept {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    auto window = glfwCreateWindow((int)extent.width, (int)extent.height, "Vulkan Graphics Demo", nullptr, nullptr);
    if (window == nullptr) {
        return std::unexpected{Error{-1, "failed to create GLFW window"}};
    }

    return WindowManager{extent, window};
}

}  // namespace vkc
