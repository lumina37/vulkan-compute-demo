#pragma once

#include <map>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/concepts.hpp"
#include "vkc/device/logical.hpp"

namespace vkc {

namespace rgs = std::ranges;

template <CSupportGetDescType... TManager>
[[nodiscard]] static inline auto genPoolSizes(const TManager&... mgrs) {
    std::map<vk::DescriptorType, int> poolSizeMap;

    const auto appendPoolSize = [&](const auto& mgr) {
        auto descType = mgr.getDescType();
        if (!poolSizeMap.contains(descType)) {
            poolSizeMap[descType] = 1;
            return;
        }
        poolSizeMap[descType]++;
    };

    (appendPoolSize(mgrs), ...);

    const auto transKV2PoolSize = [](const auto& pair) {
        auto [descType, count] = pair;
        vk::DescriptorPoolSize poolSize;
        poolSize.setType(descType);
        poolSize.setDescriptorCount(count);
        return poolSize;
    };

    std::vector<vk::DescriptorPoolSize> poolSizes =
        poolSizeMap | rgs::views::transform(transKV2PoolSize) | rgs::to<std::vector>();

    return poolSizes;
}

class DescPoolManager {
public:
    inline DescPoolManager(DeviceManager& deviceMgr, const std::span<vk::DescriptorPoolSize> poolSizes);
    inline ~DescPoolManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescPool(this Self&& self) noexcept {
        return std::forward_like<Self>(self).descPool_;
    }

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::DescriptorPool descPool_;
};

DescPoolManager::DescPoolManager(DeviceManager& deviceMgr, const std::span<vk::DescriptorPoolSize> poolSizes)
    : deviceMgr_(deviceMgr) {
    vk::DescriptorPoolCreateInfo poolInfo;
    poolInfo.setMaxSets(poolSizes.size());
    poolInfo.setPoolSizes(poolSizes);

    auto& device = deviceMgr.getDevice();
    descPool_ = device.createDescriptorPool(poolInfo);
}

DescPoolManager::~DescPoolManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyDescriptorPool(descPool_);
}

}  // namespace vkc
