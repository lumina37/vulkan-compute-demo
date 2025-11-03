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

template <CSupportGetDescType... TBox>
[[nodiscard]] static constexpr auto genPoolSizes(const TBox&... boxes) noexcept {
    std::map<vk::DescriptorType, int> poolSizeMap;

    const auto appendPoolSize = [&](const auto& box) {
        vk::DescriptorType descType = box.getDescType();
        if (!poolSizeMap.contains(descType)) {
            poolSizeMap[descType] = 1;
            return;
        }
        poolSizeMap[descType]++;
    };

    (appendPoolSize(boxes), ...);

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

template <CSupportStaticGetDescType... TBox>
[[nodiscard]] static constexpr auto genPoolSizes() noexcept {
    std::map<vk::DescriptorType, int> poolSizeMap;

    const auto appendPoolSize = [&]<CSupportStaticGetDescType T>() {
        vk::DescriptorType descType = T::getDescType();
        if (!poolSizeMap.contains(descType)) {
            poolSizeMap[descType] = 1;
            return;
        }
        poolSizeMap[descType]++;
    };

    (appendPoolSize.template operator()<TBox>(), ...);

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

class DescPoolBox {
    DescPoolBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DescriptorPool descPool) noexcept;

public:
    DescPoolBox(const DescPoolBox&) = delete;
    DescPoolBox(DescPoolBox&& rhs) noexcept;
    ~DescPoolBox() noexcept;

    [[nodiscard]] static std::expected<DescPoolBox, Error> create(
        std::shared_ptr<DeviceBox> pDeviceBox, std::span<const vk::DescriptorPoolSize> poolSizes) noexcept;

    [[nodiscard]] vk::DescriptorPool getDescPool() const noexcept { return descPool_; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::DescriptorPool descPool_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/pool.cpp"
#endif
