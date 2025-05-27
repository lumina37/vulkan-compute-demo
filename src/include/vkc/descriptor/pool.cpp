#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/pool.hpp"
#endif

namespace vkc {

DescPoolManager::DescPoolManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::DescriptorPool descPool) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), descPool_(descPool) {}

DescPoolManager::DescPoolManager(DescPoolManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)), descPool_(std::exchange(rhs.descPool_, nullptr)) {}

DescPoolManager::~DescPoolManager() noexcept {
    if (descPool_ == nullptr) return;
    vk::Device device = pDeviceMgr_->getDevice();
    device.destroyDescriptorPool(descPool_);
    descPool_ = nullptr;
}

std::expected<DescPoolManager, Error> DescPoolManager::create(
    std::shared_ptr<DeviceManager> pDeviceMgr, std::span<const vk::DescriptorPoolSize> poolSizes) noexcept {
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets((uint32_t)poolSizes.size());
    poolInfo.setPoolSizes(poolSizes);

    vk::Device device = pDeviceMgr->getDevice();
    const auto [descPoolRes, descPool] = device.createDescriptorPool(poolInfo);
    if (descPoolRes != vk::Result::eSuccess) {
        return std::unexpected{Error{descPoolRes}};
    }

    return DescPoolManager{std::move(pDeviceMgr), descPool};
}

}  // namespace vkc
