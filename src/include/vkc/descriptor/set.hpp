#pragma once

#include <array>
#include <ranges>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/concepts.hpp"
#include "vkc/descriptor/layout.hpp"
#include "vkc/descriptor/pool.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/image.hpp"

namespace vkc {

namespace rgs = std::ranges;

class DescSetManager {
public:
    inline DescSetManager(const DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                          const DescPoolManager& descPoolMgr);
    inline ~DescSetManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescSet(this Self& self) noexcept {
        return std::forward_like<Self>(self).descSet_;
    }

    template <CSupDraftWriteDescSet... TSupDraftWriteDescSet>
    inline void updateDescSets(const TSupDraftWriteDescSet&... mgrs);

private:
    const DeviceManager& deviceMgr_;      // FIXME: UAF
    const DescPoolManager& descPoolMgr_;  // FIXME: UAF
    vk::DescriptorSet descSet_;
};

DescSetManager::DescSetManager(const DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                               const DescPoolManager& descPoolMgr)
    : deviceMgr_(deviceMgr), descPoolMgr_(descPoolMgr) {
    vk::DescriptorSetAllocateInfo descSetAllocInfo;
    descSetAllocInfo.setDescriptorPool(descPoolMgr.getDescPool());
    descSetAllocInfo.setDescriptorSetCount(1);
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    descSetAllocInfo.setSetLayouts(descSetLayout);

    const auto& device = deviceMgr.getDevice();
    descSet_ = device.allocateDescriptorSets(descSetAllocInfo)[0];
}

inline DescSetManager::~DescSetManager() noexcept {
    // TODO: maybe free sth. here
}

template <CSupDraftWriteDescSet... TManager>
void DescSetManager::updateDescSets(const TManager&... mgrs) {
    const auto& device = deviceMgr_.getDevice();

    std::array writeDescSets{mgrs.draftWriteDescSet()...};
    for (auto [index, writeDescSet] : rgs::views::enumerate(writeDescSets)) {
        writeDescSet.setDstSet(getDescSet());
        writeDescSet.setDstBinding(index);
    }

    device.updateDescriptorSets(writeDescSets, nullptr);
}

}  // namespace vkc
