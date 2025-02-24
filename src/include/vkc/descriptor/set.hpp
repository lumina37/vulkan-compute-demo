#pragma once

#include <array>
#include <ranges>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/concepts.hpp"
#include "vkc/descriptor/layout.hpp"
#include "vkc/descriptor/pool.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/resource/image.hpp"

namespace vkc {

namespace rgs = std::ranges;

class DescSetManager {
public:
    inline DescSetManager(DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                          DescPoolManager& descPoolMgr);
    inline ~DescSetManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescSet(this Self&& self) noexcept {
        return std::forward_like<Self>(self).descSet_;
    }

    template <CSupportDraftWriteDescSet... TManager>
    inline void updateDescSets(const TManager&... mgrs);

private:
    DeviceManager& deviceMgr_;      // FIXME: UAF
    DescPoolManager& descPoolMgr_;  // FIXME: UAF
    vk::DescriptorSet descSet_;
};

DescSetManager::DescSetManager(DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                               DescPoolManager& descPoolMgr)
    : deviceMgr_(deviceMgr), descPoolMgr_(descPoolMgr) {
    vk::DescriptorSetAllocateInfo descSetAllocInfo;
    descSetAllocInfo.setDescriptorPool(descPoolMgr.getDescPool());
    descSetAllocInfo.setDescriptorSetCount(1);
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    descSetAllocInfo.setSetLayouts(descSetLayout);

    auto& device = deviceMgr.getDevice();
    const auto& descSets = device.allocateDescriptorSets(descSetAllocInfo);
    descSet_ = descSets[0];
}

inline DescSetManager::~DescSetManager() noexcept {
    // TODO: maybe free sth. here
}

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
