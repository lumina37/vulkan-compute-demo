#pragma once

#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "vkc/device/logical.hpp"
#include "vkc/extent.hpp"
#include "vkc/gui/surface.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource.hpp"
#include "vkc/sync/semaphore.hpp"

namespace vkc {

class SwapchainBox {
    SwapchainBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::SwapchainKHR swapchain,
                 std::vector<PresentImageBox>&& imageBoxs) noexcept;

public:
    SwapchainBox(const SwapchainBox&) = delete;
    SwapchainBox(SwapchainBox&& rhs) noexcept;
    ~SwapchainBox() noexcept;

    [[nodiscard]] static std::expected<SwapchainBox, Error> create(
        std::shared_ptr<DeviceBox> pDeviceBox, SurfaceBox& surfaceBox, std::span<const uint32_t> queueFamilyIndices,
        const Extent& extent, vk::ColorSpaceKHR colorspace = vk::ColorSpaceKHR::eSrgbNonlinear) noexcept;

    [[nodiscard]] vk::SwapchainKHR getSwapchain() const noexcept { return swapchain_; }

    template <typename Self>
    [[nodiscard]] auto&& getPresentImageBox(this Self&& self, uint32_t i) noexcept {
        return std::forward_like<Self>(self).presentImageBoxs_[i];
    }

    [[nodiscard]] std::expected<uint32_t, Error> acquireImageIndex(
        SemaphoreBox& signalSemaphoreBox, uint64_t timeout = std::numeric_limits<uint64_t>::max()) noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::SwapchainKHR swapchain_;
    std::vector<PresentImageBox> presentImageBoxs_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/swapchain.cpp"
#endif
