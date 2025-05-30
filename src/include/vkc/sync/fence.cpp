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

FenceManager::FenceManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Fence fence) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), fence_(fence) {}

FenceManager::FenceManager(FenceManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)), fence_(std::exchange(rhs.fence_, nullptr)) {}

FenceManager::~FenceManager() noexcept {
    if (fence_ == nullptr) return;
    vk::Device device = pDeviceMgr_->getDevice();
    device.destroyFence(fence_);
    fence_ = nullptr;
}

std::expected<FenceManager, Error> FenceManager::create(std::shared_ptr<DeviceManager> pDeviceMgr) noexcept {
    vk::Device device = pDeviceMgr->getDevice();
    vk::FenceCreateInfo fenceInfo;
    const auto [fenceRes, fence] = device.createFence(fenceInfo);
    if (fenceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{fenceRes}};
    }

    return FenceManager{std::move(pDeviceMgr), fence};
}

std::expected<void, Error> FenceManager::wait(uint64_t timeout) noexcept {
    vk::Device device = pDeviceMgr_->getDevice();
    const auto waitFenceRes = device.waitForFences(fence_, true, timeout);
    if (waitFenceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{waitFenceRes}};
    }

    return {};
}

std::expected<void, Error> FenceManager::reset() noexcept {
    vk::Device device = pDeviceMgr_->getDevice();
    const auto resetFenceRes = device.resetFences(fence_);
    if (resetFenceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{resetFenceRes}};
    }

    return {};
}

}  // namespace vkc
