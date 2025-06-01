#include <cstdint>
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

SwapChainManager::SwapChainManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::SwapchainKHR swapchain,
                                   std::vector<PresentImageManager>&& imageMgrs) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), swapchain_(swapchain), imageMgrs_(std::move(imageMgrs)) {}

SwapChainManager::SwapChainManager(SwapChainManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      swapchain_(std::exchange(rhs.swapchain_, nullptr)),
      imageMgrs_(std::move(rhs.imageMgrs_)) {}

SwapChainManager::~SwapChainManager() noexcept {
    if (swapchain_ == nullptr) return;
    auto device = pDeviceMgr_->getDevice();
    device.destroySwapchainKHR(swapchain_);
    swapchain_ = nullptr;
}

std::expected<SwapChainManager, Error> SwapChainManager::create(PhyDeviceManager& phyDeviceMgr,
                                                                std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                SurfaceManager& surfaceMgr,
                                                                const std::span<const uint32_t> queueFamilyIndices,
                                                                const Extent& extent,
                                                                const vk::ColorSpaceKHR colorspace) noexcept {
    vk::SwapchainCreateInfoKHR swapchainInfo;
    swapchainInfo.setSurface(surfaceMgr.getSurface());
    swapchainInfo.setMinImageCount(2);
    swapchainInfo.setImageFormat(extent.format());
    swapchainInfo.setImageColorSpace(colorspace);
    swapchainInfo.setImageExtent(extent.extent());
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst);
    swapchainInfo.setPresentMode(vk::PresentModeKHR::eFifo);
    swapchainInfo.setClipped(true);

    if (queueFamilyIndices.size() == 1) {
        swapchainInfo.setImageSharingMode(vk::SharingMode::eExclusive);
    } else {
        swapchainInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
        swapchainInfo.setQueueFamilyIndices(queueFamilyIndices);
    }

    auto device = pDeviceMgr->getDevice();
    auto swapchainRes = device.createSwapchainKHR(swapchainInfo);
    if (swapchainRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{swapchainRes.result}};
    }
    vk::SwapchainKHR swapchain = swapchainRes.value;

    auto imagesRes = device.getSwapchainImagesKHR(swapchain);
    if (imagesRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{imagesRes.result}};
    }
    auto images = std::move(imagesRes.value);

    std::vector<PresentImageManager> imageMgrs;
    imageMgrs.reserve(images.size());

    for (auto image : images) {
        auto imageMgrRes = PresentImageManager::create(phyDeviceMgr, pDeviceMgr, image, extent);
        if (!imageMgrRes) return std::unexpected{std::move(imageMgrRes.error())};
        auto& imageMgr = imageMgrRes.value();
        imageMgrs.emplace_back(std::move(imageMgr));
    }

    return SwapChainManager{std::move(pDeviceMgr), swapchain, std::move(imageMgrs)};
}

std::expected<uint32_t, Error> SwapChainManager::acquireImageIndex(SemaphoreManager& signalSemaphoreMgr,
                                                                   uint64_t timeout) noexcept {
    auto signalSemaphore = signalSemaphoreMgr.getSemaphore();
    auto device = pDeviceMgr_->getDevice();
    const auto nextImageIndexRes = device.acquireNextImageKHR(swapchain_, timeout, signalSemaphore);
    if (nextImageIndexRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{nextImageIndexRes.result}};
    }
    auto nextImageIndex = nextImageIndexRes.value;

    return nextImageIndex;
}

std::expected<void, Error> SwapChainManager::present(QueueManager& queueMgr, uint32_t imageIndex) noexcept {
    vk::PresentInfoKHR presentInfo;
    presentInfo.setImageIndices(imageIndex);
    presentInfo.setSwapchains(swapchain_);

    auto presentQueue = queueMgr.getQueue();
    const auto presentRes = presentQueue.presentKHR(presentInfo);
    if (presentRes != vk::Result::eSuccess) {
        return std::unexpected{Error{presentRes}};
    }

    return {};
}

}  // namespace vkc
