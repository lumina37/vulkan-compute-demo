#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/pool.hpp"
#endif

namespace vkc {

DescPoolManager::DescPoolManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::DescriptorPool descPool) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), descPool_(descPool) {}

DescPoolManager::DescPoolManager(DescPoolManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)), descPool_(std::exchange(rhs.descPool_, nullptr)) {}

DescPoolManager::~DescPoolManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    if (descPool_ == nullptr) return;
    device.destroyDescriptorPool(descPool_);
    descPool_ = nullptr;
}

std::expected<DescPoolManager, Error> DescPoolManager::create(
    std::shared_ptr<DeviceManager> pDeviceMgr, std::span<const vk::DescriptorPoolSize> poolSizes) noexcept {
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets((uint32_t)poolSizes.size());
    poolInfo.setPoolSizes(poolSizes);

    auto& device = pDeviceMgr->getDevice();
    vk::DescriptorPool descPool = device.createDescriptorPool(poolInfo);

    return DescPoolManager{std::move(pDeviceMgr), descPool};
}

}  // namespace vkc
