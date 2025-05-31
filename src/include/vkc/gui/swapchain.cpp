#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/gui/surface.hpp"
#include "vkc/gui/swapchain.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/swapchain.hpp"
#endif

namespace vkc {

SwapChainManager::SwapChainManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::SwapchainKHR swapchain) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), swapchain_(swapchain) {}

SwapChainManager::SwapChainManager(SwapChainManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)), swapchain_(std::exchange(rhs.swapchain_, nullptr)) {}

SwapChainManager::~SwapChainManager() noexcept {
    if (swapchain_ == nullptr) return;
    auto device = pDeviceMgr_->getDevice();
    device.destroySwapchainKHR(swapchain_);
}

std::expected<SwapChainManager, Error> SwapChainManager::create(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                SurfaceManager& surfaceMgr,
                                                                std::span<uint32_t> queueFamilyIndices,
                                                                const Extent& extent) noexcept {
    vk::SwapchainCreateInfoKHR swapchainInfo;
    swapchainInfo.setSurface(surfaceMgr.getSurface());
    swapchainInfo.setMinImageCount(2);
    swapchainInfo.setImageFormat(extent.format());
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);  // TODO: auto-select
    swapchainInfo.setImageExtent(extent.extent());
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
    swapchainInfo.setPresentMode(vk::PresentModeKHR::eFifo);
    swapchainInfo.setClipped(true);

    if (queueFamilyIndices.size() == 1) {
        swapchainInfo.setImageSharingMode(vk::SharingMode::eExclusive);
    } else {
        swapchainInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
        swapchainInfo.setQueueFamilyIndices(queueFamilyIndices);
    }

    auto device = pDeviceMgr->getDevice();
    auto swapchainResult = device.createSwapchainKHR(swapchainInfo);
    if (swapchainResult.result != vk::Result::eSuccess) {
        return std::unexpected{Error{(int)swapchainResult.result, "failed to create swapchain"}};
    }
    vk::SwapchainKHR swapchain = swapchainResult.value;

    return SwapChainManager{std::move(pDeviceMgr), swapchain};
}

}  // namespace vkc
