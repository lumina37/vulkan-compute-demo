#pragma once

#include <expected>
#include <vector>

#include <GLFW/glfw3.h>

#include "vkc/helper/error.hpp"

namespace vkc {

class WindowBox {
    WindowBox(vk::Extent2D extent, GLFWwindow* window) noexcept;

public:
    WindowBox(const WindowBox&) = delete;
    WindowBox(WindowBox&& rhs) noexcept;
    ~WindowBox() noexcept;

    static std::expected<void, Error> globalInit() noexcept;
    static void globalDestroy() noexcept;

    [[nodiscard]] static std::expected<std::vector<std::string_view>, Error> getExtensions();

    [[nodiscard]] static std::expected<WindowBox, Error> create(vk::Extent2D extent) noexcept;

    [[nodiscard]] vk::Extent2D getExtent() const noexcept { return extent_; }
    [[nodiscard]] GLFWwindow* getWindow() const noexcept { return window_; }

private:
    vk::Extent2D extent_;
    GLFWwindow* window_;
};

}  // namespace vkc
