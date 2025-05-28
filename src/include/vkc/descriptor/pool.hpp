#pragma once

#include <expected>
#include <map>
#include <memory>
#include <ranges>
#include <span>
#include <vector>

#include "vkc/descriptor/concepts.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

namespace rgs = std::ranges;

template <CSupportGetDescType... TManager>
[[nodiscard]] static constexpr auto genPoolSizes(const TManager&... mgrs) noexcept {
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

    constexpr auto transKV2PoolSize = [](const auto& pair) {
        auto [descType, count] = pair;
        vk::DescriptorPoolSize poolSize;
        poolSize.setType(descType);
        poolSize.setDescriptorCount(count);
        return poolSize;
    };

    const std::vector<vk::DescriptorPoolSize> poolSizes =
        poolSizeMap | rgs::views::transform(transKV2PoolSize) | rgs::to<std::vector>();

    return poolSizes;
}

class DescPoolManager {
    DescPoolManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::DescriptorPool descPool) noexcept;

public:
    DescPoolManager(DescPoolManager&& rhs) noexcept;
    ~DescPoolManager() noexcept;

    [[nodiscard]] static std::expected<DescPoolManager, Error> create(
        std::shared_ptr<DeviceManager> pDeviceMgr, std::span<const vk::DescriptorPoolSize> poolSizes) noexcept;

    [[nodiscard]] vk::DescriptorPool getDescPool() const noexcept { return descPool_; }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::DescriptorPool descPool_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/pool.cpp"
#endif
