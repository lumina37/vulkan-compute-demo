#pragma once

#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/extent.hpp"
#include "vkc/gui/surface.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/resource/present_image.hpp"
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
    [[nodiscard]] auto&& getImageBox(this Self&& self, uint32_t i) noexcept {
        return std::forward_like<Self>(self).imageBoxs_[i];
    }

    [[nodiscard]] std::expected<uint32_t, Error> acquireImageIndex(
        SemaphoreBox& signalSemaphoreBox, uint64_t timeout = std::numeric_limits<uint64_t>::max()) noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::SwapchainKHR swapchain_;
    std::vector<PresentImageBox> imageBoxs_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/swapchain.cpp"
#endif
