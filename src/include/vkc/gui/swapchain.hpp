#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/extent.hpp"
#include "vkc/gui/surface.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

class SwapChainManager {
    SwapChainManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::SwapchainKHR swapchain) noexcept;

public:
    SwapChainManager(SwapChainManager&& rhs) noexcept;
    ~SwapChainManager() noexcept;

    [[nodiscard]] static std::expected<SwapChainManager, Error> create(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                       SurfaceManager& surfaceMgr,
                                                                       std::span<uint32_t> queueFamilyIndices,
                                                                       const Extent& extent) noexcept;

    [[nodiscard]] vk::SwapchainKHR getSwapchain() const noexcept { return swapchain_; }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::SwapchainKHR swapchain_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/swapchain.cpp"
#endif
