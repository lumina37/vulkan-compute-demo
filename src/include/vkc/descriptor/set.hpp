#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "vkc/descriptor/concepts.hpp"
#include "vkc/descriptor/layout.hpp"
#include "vkc/descriptor/pool.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <CSupportDraftWriteDescSet... TBox>
[[nodiscard]] static constexpr auto genWriteDescSets(const TBox&... boxes) noexcept {
    constexpr auto genWriteDescSet = [](const auto& box, const size_t index) {
        vk::WriteDescriptorSet writeDescSet = box.draftWriteDescSet();
        writeDescSet.setDstBinding((uint32_t)index);
        return writeDescSet;
    };

    const auto genWriteDescSetHelper = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{genWriteDescSet(boxes, Is)...};
    };

    return genWriteDescSetHelper(std::index_sequence_for<TBox...>{});
}

class DescSetsBox {
    DescSetsBox(std::shared_ptr<DeviceBox>&& pDeviceBox, std::vector<vk::DescriptorSet>&& descSets) noexcept;

public:
    using TDescSetLayoutBoxCRef = std::reference_wrapper<const DescSetLayoutBox>;

    DescSetsBox(const DescSetsBox&) = delete;
    DescSetsBox(DescSetsBox&&) noexcept = default;

    [[nodiscard]] static std::expected<DescSetsBox, Error> create(
        std::shared_ptr<DeviceBox> pDeviceBox, DescPoolBox& descPoolBox,
        std::span<const TDescSetLayoutBoxCRef> descSetLayoutBoxCRefs) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescSets(this Self&& self) noexcept {
        return std::forward_like<Self>(self).descSets_;
    }

    [[nodiscard]] vk::DescriptorSet getDescSet(const int index) const noexcept { return descSets_[index]; }

    void updateDescSets(std::span<const std::span<const vk::WriteDescriptorSet>> writeDescSetTemplatesRefs) noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    std::vector<vk::DescriptorSet> descSets_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/set.cpp"
#endif
