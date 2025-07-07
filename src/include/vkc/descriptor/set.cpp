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

DescSetsBox::DescSetsBox(std::shared_ptr<DeviceBox>&& pDeviceBox,
                                 std::vector<vk::DescriptorSet>&& descSets) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), descSets_(std::move(descSets)) {}

std::expected<DescSetsBox, Error> DescSetsBox::create(
    std::shared_ptr<DeviceBox> pDeviceBox, DescPoolBox& descPoolBox,
    std::span<const TDescSetLayoutBoxCRef> descSetLayoutBoxCRefs) noexcept {
    const auto genDescSetLayout = [](const TDescSetLayoutBoxCRef& boxRef) {
        const DescSetLayoutBox& descSetLayoutBox = boxRef.get();
        vk::DescriptorSetLayout descSetLayout = descSetLayoutBox.getDescSetLayout();
        return descSetLayout;
    };

    const auto descSetLayouts =
        descSetLayoutBoxCRefs | rgs::views::transform(genDescSetLayout) | rgs::to<std::vector>();

    vk::DescriptorSetAllocateInfo descSetAllocInfo;
    vk::DescriptorPool descPool = descPoolBox.getDescPool();
    descSetAllocInfo.setDescriptorPool(descPool);
    descSetAllocInfo.setDescriptorSetCount((uint32_t)descSetLayoutBoxCRefs.size());
    descSetAllocInfo.setSetLayouts(descSetLayouts);

    vk::Device device = pDeviceBox->getDevice();
    auto [descSetsRes, descSets] = device.allocateDescriptorSets(descSetAllocInfo);
    if (descSetsRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, descSetsRes}};
    }

    return DescSetsBox{std::move(pDeviceBox), std::move(descSets)};
}

void DescSetsBox::updateDescSets(
    std::span<const std::span<const vk::WriteDescriptorSet>> writeDescSetTemplatesRefs) noexcept {
    int writeDescSetCount = 0;
    for (const auto& writeDescSetTemplates : writeDescSetTemplatesRefs) {
        writeDescSetCount += (int)writeDescSetTemplates.size();
    }

    std::vector<vk::WriteDescriptorSet> writeDescSets;
    writeDescSets.reserve(writeDescSetCount);

    for (const auto& [idx, writeDescSetTemplates] : rgs::views::enumerate(writeDescSetTemplatesRefs)) {
        vk::DescriptorSet descSet = descSets_[idx];
        for (const auto& writeDescSetTemplate : writeDescSetTemplates) {
            vk::WriteDescriptorSet writeDescSet = writeDescSetTemplate;
            writeDescSet.setDstSet(descSet);
            writeDescSets.emplace_back(writeDescSet);
        }
    }

    vk::Device device = pDeviceBox_->getDevice();
    device.updateDescriptorSets(writeDescSets, nullptr);
}

}  // namespace vkc
