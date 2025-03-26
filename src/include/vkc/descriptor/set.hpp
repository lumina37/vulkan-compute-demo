#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/concepts.hpp"
#include "vkc/descriptor/layout.hpp"
#include "vkc/descriptor/pool.hpp"
#include "vkc/device/logical.hpp"

namespace vkc {

class DescSetManager {
public:
    DescSetManager(DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                   DescPoolManager& descPoolMgr);

    template <typename Self>
    [[nodiscard]] auto&& getDescSet(this Self&& self) noexcept {
        return std::forward_like<Self>(self).descSet_;
    }

    template <CSupportDraftWriteDescSet... TManager>
    void updateDescSets(const TManager&... mgrs);

private:
    DeviceManager& deviceMgr_;      // FIXME: UAF
    DescPoolManager& descPoolMgr_;  // FIXME: UAF
    vk::DescriptorSet descSet_;
};

template <CSupportDraftWriteDescSet... TManager>
void DescSetManager::updateDescSets(const TManager&... mgrs) {
    const auto genWriteDescSet = [this](const auto& mgr, size_t index) {
        vk::WriteDescriptorSet writeDescSet = mgr.draftWriteDescSet();
        writeDescSet.setDstSet(getDescSet());
        writeDescSet.setDstBinding(index);
        return writeDescSet;
    };

    const auto genWriteDescSetHelper = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array{genWriteDescSet(mgrs, Is)...};
    };

    const std::array writeDescSets{genWriteDescSetHelper(std::index_sequence_for<TManager...>{})};

    auto& device = deviceMgr_.getDevice();
    device.updateDescriptorSets(writeDescSets, nullptr);
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/set.cpp"
#endif
