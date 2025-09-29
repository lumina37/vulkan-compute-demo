#include <cstdint>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/gui/surface.hpp"
#include "vkc/gui/swapchain.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/resource.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/swapchain.hpp"
#endif

namespace vkc {

SwapchainBox::SwapchainBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::SwapchainKHR swapchain,
                           std::vector<PresentImageBox>&& imageBoxs) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), swapchain_(swapchain), presentImageBoxs_(std::move(imageBoxs)) {}

SwapchainBox::SwapchainBox(SwapchainBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      swapchain_(std::exchange(rhs.swapchain_, nullptr)),
      presentImageBoxs_(std::move(rhs.presentImageBoxs_)) {}

SwapchainBox::~SwapchainBox() noexcept {
    if (swapchain_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroySwapchainKHR(swapchain_);
    swapchain_ = nullptr;
}

std::expected<SwapchainBox, Error> SwapchainBox::create(std::shared_ptr<DeviceBox> pDeviceBox, SurfaceBox& surfaceBox,
                                                        const std::span<const uint32_t> queueFamilyIndices,
                                                        const Extent& extent,
                                                        const vk::ColorSpaceKHR colorspace) noexcept {
    vk::SwapchainCreateInfoKHR swapchainInfo;
    swapchainInfo.setSurface(surfaceBox.getSurface());
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

    vk::Device device = pDeviceBox->getDevice();
    auto swapchainRes = device.createSwapchainKHR(swapchainInfo);
    if (swapchainRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, swapchainRes.result}};
    }
    vk::SwapchainKHR swapchain = swapchainRes.value;

    auto imagesRes = device.getSwapchainImagesKHR(swapchain);
    if (imagesRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, imagesRes.result}};
    }
    auto images = std::move(imagesRes.value);

    std::vector<PresentImageBox> imageBoxs;
    imageBoxs.reserve(images.size());

    for (auto image : images) {
        ImageBox imageBox = ImageBox::createWithoutOwning(image, extent);
        auto presentImageBoxRes = PresentImageBox::create(pDeviceBox, imageBox);
        if (!presentImageBoxRes) return std::unexpected{std::move(presentImageBoxRes.error())};
        auto& presentImageBox = presentImageBoxRes.value();
        imageBoxs.emplace_back(std::move(presentImageBox));
    }

    return SwapchainBox{std::move(pDeviceBox), swapchain, std::move(imageBoxs)};
}

std::expected<uint32_t, Error> SwapchainBox::acquireImageIndex(SemaphoreBox& signalSemaphoreBox,
                                                               uint64_t timeout) noexcept {
    const vk::Semaphore signalSemaphore = signalSemaphoreBox.getSemaphore();
    const vk::Device device = pDeviceBox_->getDevice();
    const auto imageIndexRes = device.acquireNextImageKHR(swapchain_, timeout, signalSemaphore);
    if (imageIndexRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, imageIndexRes.result}};
    }
    const uint32_t imageIndex = imageIndexRes.value;

    return imageIndex;
}

}  // namespace vkc
