#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/window.hpp"
#endif

namespace vkc {

WindowManager::WindowManager(vk::Extent2D extent, GLFWwindow* window) noexcept : extent_(extent), window_(window) {}

WindowManager::WindowManager(WindowManager&& rhs) noexcept
    : extent_(rhs.extent_), window_(std::exchange(rhs.window_, nullptr)) {}

WindowManager::~WindowManager() noexcept {
    if (window_ == nullptr) return;
    glfwDestroyWindow(window_);
}

std::pair<uint32_t, const char**> WindowManager::getExtensions() {
    uint32_t count;
    const auto pExts = glfwGetRequiredInstanceExtensions(&count);
    return {count, pExts};
}

std::expected<WindowManager, Error> WindowManager::create(const vk::Extent2D extent) noexcept {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    auto window = glfwCreateWindow((int)extent.width, (int)extent.height, "Vulkan Graphics Demo", nullptr, nullptr);
    if (window == nullptr) {
        return std::unexpected{Error{-1, "failed to create swapchain"}};
    }

    return WindowManager{extent, window};
}

}  // namespace vkc
