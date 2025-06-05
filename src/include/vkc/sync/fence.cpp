#include <cstdint>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/sync/fence.hpp"
#endif

namespace vkc {

FenceBox::FenceBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Fence fence) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), fence_(fence) {}

FenceBox::FenceBox(FenceBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)), fence_(std::exchange(rhs.fence_, nullptr)) {}

FenceBox::~FenceBox() noexcept {
    if (fence_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyFence(fence_);
    fence_ = nullptr;
}

std::expected<FenceBox, Error> FenceBox::create(std::shared_ptr<DeviceBox> pDeviceBox) noexcept {
    vk::Device device = pDeviceBox->getDevice();
    vk::FenceCreateInfo fenceInfo;
    const auto [fenceRes, fence] = device.createFence(fenceInfo);
    if (fenceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{fenceRes}};
    }

    return FenceBox{std::move(pDeviceBox), fence};
}

std::expected<void, Error> FenceBox::wait(uint64_t timeout) noexcept {
    vk::Device device = pDeviceBox_->getDevice();
    const auto waitFenceRes = device.waitForFences(fence_, true, timeout);
    if (waitFenceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{waitFenceRes}};
    }

    return {};
}

std::expected<void, Error> FenceBox::reset() noexcept {
    vk::Device device = pDeviceBox_->getDevice();
    const auto resetFenceRes = device.resetFences(fence_);
    if (resetFenceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{resetFenceRes}};
    }

    return {};
}

}  // namespace vkc
