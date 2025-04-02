#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/concepts.hpp"
#include "vkc/descriptor/layout.hpp"
#include "vkc/descriptor/pool.hpp"
#include "vkc/device/logical.hpp"

namespace vkc {

template <CSupportDraftWriteDescSet... TManager>
[[nodiscard]] static constexpr inline auto genWriteDescSets(const TManager&... mgrs) {
    const auto genWriteDescSet = [](const auto& mgr, const size_t index) {
        vk::WriteDescriptorSet writeDescSet = mgr.draftWriteDescSet();
        writeDescSet.setDstBinding(index);
        return writeDescSet;
    };

    const auto genWriteDescSetHelper = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{genWriteDescSet(mgrs, Is)...};
    };

    return genWriteDescSetHelper(std::index_sequence_for<TManager...>{});
}

class DescSetsManager {
public:
    using TDescSetLayoutMgrCRef = std::reference_wrapper<const DescSetLayoutManager>;
    DescSetsManager(const std::shared_ptr<DeviceManager>& pDeviceMgr, DescPoolManager& descPoolMgr,
                    std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs);

    template <typename Self>
    [[nodiscard]] auto&& getDescSets(this Self&& self) noexcept {
        return std::forward_like<Self>(self).descSets_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getDescSet(this Self&& self, const int index) noexcept {
        return std::forward_like<Self>(self).descSets_[index];
    }

    void updateDescSets(std::span<const std::span<const vk::WriteDescriptorSet>> writeDescSetTemplatesRefs);

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    std::vector<vk::DescriptorSet> descSets_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/set.cpp"
#endif
