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
    vk::DescriptorPoolSize samplerPoolSize;
    samplerPoolSize.setType(vk::DescriptorType::eSampler);
    samplerPoolSize.setDescriptorCount(4);
    vk::DescriptorPoolSize imagePoolSize;
    imagePoolSize.setType(vk::DescriptorType::eStorageImage);
    imagePoolSize.setDescriptorCount(4);
    vk::DescriptorPoolSize texturePoolSize;
    texturePoolSize.setType(vk::DescriptorType::eSampledImage);
    texturePoolSize.setDescriptorCount(4);
    const std::array poolSizes{samplerPoolSize, imagePoolSize, texturePoolSize};

    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets(1);
    poolInfo.setPoolSizes(poolSizes);

    const auto& device = deviceMgr.getDevice();
    descPool_ = device.createDescriptorPool(poolInfo);
}

DescPoolManager::~DescPoolManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyDescriptorPool(descPool_);
}

}  // namespace vkc
