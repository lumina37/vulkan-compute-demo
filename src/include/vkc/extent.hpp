#pragma once

#include <cstddef>
#include <utility>

#include <vulkan/vulkan.hpp>

namespace vkc {

class ExtentManager {
public:
    inline ExtentManager(const int width, const int height, const int comps) : extent_(width, height), comps_(comps) {}

    [[nodiscard]] inline int width() const noexcept { return extent_.width; }
    [[nodiscard]] inline int height() const noexcept { return extent_.height; }
    [[nodiscard]] inline int comps() const noexcept { return comps_; }
    [[nodiscard]] inline size_t size() const noexcept { return extent_.width * extent_.height * comps_; }
    [[nodiscard]] inline vk::Extent2D extent() const noexcept { return extent_; }
    [[nodiscard]] inline vk::Extent3D extent3D() const noexcept { return {extent_.width, extent_.height, 1}; }
    [[nodiscard]] inline vk::Format formatUnorm() const noexcept;

private:
    vk::Extent2D extent_;
    int comps_;
};

vk::Format ExtentManager::formatUnorm() const noexcept {
    switch (comps_) {
        case 1:
            return vk::Format::eR8Unorm;
        case 2:
            return vk::Format::eR8G8Unorm;
        case 3:
            return vk::Format::eR8G8B8Unorm;
        case 4:
            return vk::Format::eR8G8B8A8Unorm;
        default:
            std::unreachable();
    }
}

}  // namespace vkc
