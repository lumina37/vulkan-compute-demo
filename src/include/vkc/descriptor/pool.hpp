#pragma once

#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

namespace vkc {

class DescPoolManager {
public:
    inline DescPoolManager(const DeviceManager& deviceMgr);
    inline ~DescPoolManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescPool(this Self& self) noexcept {
        return std::forward_like<Self>(self).descPool_;
    }

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::DescriptorPool descPool_;
};

DescPoolManager::DescPoolManager(const DeviceManager& deviceMgr) : deviceMgr_(deviceMgr) {
    vk::DescriptorPoolSize poolSize;
    poolSize.setType(vk::DescriptorType::eStorageImage);
    poolSize.setDescriptorCount(2);

    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets(1);
    poolInfo.setPoolSizes(poolSize);

    const auto& device = deviceMgr.getDevice();
    descPool_ = device.createDescriptorPool(poolInfo);
}

DescPoolManager::~DescPoolManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyDescriptorPool(descPool_);
}

}  // namespace vkc
