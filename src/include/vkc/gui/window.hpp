#pragma once

#include <expected>
#include <span>

#include <GLFW/glfw3.h>

#include "vkc/helper/error.hpp"

namespace vkc {

class WindowManager {
    WindowManager(vk::Extent2D extent, GLFWwindow* window) noexcept;

public:
    WindowManager(WindowManager&& rhs) noexcept;
    ~WindowManager() noexcept;

    static std::expected<void, Error> globalInit() noexcept;
    static void globalDestroy() noexcept;

    [[nodiscard]] static std::expected<std::span<const char*>, Error> getExtensions();

    [[nodiscard]] static std::expected<WindowManager, Error> create(vk::Extent2D extent) noexcept;

    [[nodiscard]] vk::Extent2D getExtent() const noexcept { return extent_; }
    [[nodiscard]] GLFWwindow* getWindow() const noexcept { return window_; }

private:
    vk::Extent2D extent_;
    GLFWwindow* window_;
};

}  // namespace vkc
