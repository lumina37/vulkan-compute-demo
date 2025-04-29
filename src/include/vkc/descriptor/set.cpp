#include <cstdint>
#include <expected>
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

#include "vkc/descriptor/layout.hpp"
#include "vkc/descriptor/pool.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/set.hpp"
#endif

namespace vkc {

DescSetsManager::DescSetsManager(std::shared_ptr<DeviceManager>&& pDeviceMgr,
                                 std::vector<vk::DescriptorSet>&& descSets) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), descSets_(std::move(descSets)) {}

std::expected<DescSetsManager, Error> DescSetsManager::create(
    std::shared_ptr<DeviceManager> pDeviceMgr, DescPoolManager& descPoolMgr,
    std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs) noexcept {
    const auto genDescSetLayout = [](const TDescSetLayoutMgrCRef& mgrRef) {
        const auto& descSetLayoutMgr = mgrRef.get();
        const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
        return descSetLayout;
    };

    const auto descSetLayouts =
        descSetLayoutMgrCRefs | rgs::views::transform(genDescSetLayout) | rgs::to<std::vector>();

    vk::DescriptorSetAllocateInfo descSetAllocInfo;
    auto& descPool = descPoolMgr.getDescPool();
    descSetAllocInfo.setDescriptorPool(descPool);
    descSetAllocInfo.setDescriptorSetCount((uint32_t)descSetLayoutMgrCRefs.size());
    descSetAllocInfo.setSetLayouts(descSetLayouts);

    auto& device = pDeviceMgr->getDevice();
    auto [descSetsRes, descSets] = device.allocateDescriptorSets(descSetAllocInfo);
    if (descSetsRes != vk::Result::eSuccess) {
        return std::unexpected{Error{descSetsRes}};
    }

    return DescSetsManager{std::move(pDeviceMgr), std::move(descSets)};
}

void DescSetsManager::updateDescSets(
    std::span<const std::span<const vk::WriteDescriptorSet>> writeDescSetTemplatesRefs) noexcept {
    int writeDescSetCount = 0;
    for (const auto& writeDescSetTemplates : writeDescSetTemplatesRefs) {
        writeDescSetCount += (int)writeDescSetTemplates.size();
    }

    std::vector<vk::WriteDescriptorSet> writeDescSets;
    writeDescSets.reserve(writeDescSetCount);

    for (const auto& [idx, writeDescSetTemplates] : rgs::views::enumerate(writeDescSetTemplatesRefs)) {
        const auto& descSet = descSets_[idx];
        for (const auto& writeDescSetTemplate : writeDescSetTemplates) {
            vk::WriteDescriptorSet writeDescSet = writeDescSetTemplate;
            writeDescSet.setDstSet(descSet);
            writeDescSets.emplace_back(writeDescSet);
        }
    }

    auto& device = pDeviceMgr_->getDevice();
    device.updateDescriptorSets(writeDescSets, nullptr);
}

}  // namespace vkc
