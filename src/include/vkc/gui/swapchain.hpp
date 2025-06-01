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
#include "vkc/device/queue.hpp"
#include "vkc/extent.hpp"
#include "vkc/gui/surface.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/resource/present_image.hpp"
#include "vkc/sync/semaphore.hpp"

namespace vkc {

class SwapChainManager {
    SwapChainManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::SwapchainKHR swapchain,
                     std::vector<PresentImageManager>&& imageMgrs) noexcept;

public:
    SwapChainManager(SwapChainManager&& rhs) noexcept;
    ~SwapChainManager() noexcept;

    [[nodiscard]] static std::expected<SwapChainManager, Error> create(
        PhyDeviceManager& phyDeviceMgr, std::shared_ptr<DeviceManager> pDeviceMgr, SurfaceManager& surfaceMgr,
        std::span<const uint32_t> queueFamilyIndices, const Extent& extent,
        vk::ColorSpaceKHR colorspace = vk::ColorSpaceKHR::eSrgbNonlinear) noexcept;

    [[nodiscard]] vk::SwapchainKHR getSwapchain() const noexcept { return swapchain_; }

    template <typename Self>
    [[nodiscard]] auto&& getImageMgr(this Self&& self, int i) noexcept {
        return std::forward_like<Self>(self).imageMgrs_[i];
    }

    [[nodiscard]] std::expected<uint32_t, Error> acquireImageIndex(
        SemaphoreManager& signalSemaphoreMgr, uint64_t timeout = std::numeric_limits<uint64_t>::max()) noexcept;
    [[nodiscard]] std::expected<void, Error> present(QueueManager& queueMgr, uint32_t imageIndex) noexcept;

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::SwapchainKHR swapchain_;
    std::vector<PresentImageManager> imageMgrs_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/swapchain.cpp"
#endif
