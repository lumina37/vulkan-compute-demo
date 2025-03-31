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
    DescPoolManager(const std::shared_ptr<DeviceManager>& pDeviceMgr, std::span<const vk::DescriptorPoolSize> poolSizes);
    ~DescPoolManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescPool(this Self&& self) noexcept {
        return std::forward_like<Self>(self).descPool_;
    }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::DescriptorPool descPool_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/pool.cpp"
#endif
