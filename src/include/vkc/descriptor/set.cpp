#include <memory>
#include <ranges>
#include <span>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/layout.hpp"
#include "vkc/descriptor/pool.hpp"
#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/set.hpp"
#endif

namespace vkc {

DescSetsManager::DescSetsManager(const std::shared_ptr<DeviceManager>& pDeviceMgr, DescPoolManager& descPoolMgr,
                                 std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs)
    : pDeviceMgr_(pDeviceMgr) {
    const auto descSetLayouts = descSetLayoutMgrCRefs | rgs::views::transform([](const TDescSetLayoutMgrCRef& mgrRef) {
                                    const auto& descSetLayoutMgr = mgrRef.get();
                                    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
                                    return descSetLayout;
                                }) |
                                rgs::to<std::vector>();

    vk::DescriptorSetAllocateInfo descSetAllocInfo;
    auto& descPool = descPoolMgr.getDescPool();
    descSetAllocInfo.setDescriptorPool(descPool);
    descSetAllocInfo.setDescriptorSetCount(descSetLayoutMgrCRefs.size());
    descSetAllocInfo.setSetLayouts(descSetLayouts);

    auto& device = pDeviceMgr->getDevice();
    descSets_ = device.allocateDescriptorSets(descSetAllocInfo);
}

void DescSetsManager::updateDescSets(std::span<const std::span<const vk::WriteDescriptorSet>> writeDescSetTemplatesRefs) {
    int writeDescSetCount = 0;
    for (const auto& writeDescSetTemplates : writeDescSetTemplatesRefs) {
        writeDescSetCount += writeDescSetTemplates.size();
    }

    std::vector<vk::WriteDescriptorSet> writeDescSets;
    writeDescSets.reserve(writeDescSetCount);

    for (const auto& [idx, writeDescSetTemplates] : rgs::views::enumerate(writeDescSetTemplatesRefs)) {
        const auto& descSet = descSets_[idx];
        for (const auto& writeDescSetTemplate : writeDescSetTemplates) {
            vk::WriteDescriptorSet writeDescSet = writeDescSetTemplate;
            writeDescSet.setDstSet(descSet);
            writeDescSets.emplace_back(std::move(writeDescSet));
        }
    }

    auto& device = pDeviceMgr_->getDevice();
    device.updateDescriptorSets(writeDescSets, nullptr);
}

}  // namespace vkc
