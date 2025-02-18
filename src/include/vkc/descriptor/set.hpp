#pragma once

#include <array>
#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/layout.hpp"
#include "vkc/descriptor/pool.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/image.hpp"

namespace vkc {

class DescSetManager {
public:
    inline DescSetManager(const DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                          const DescPoolManager& descPoolMgr);
    inline ~DescSetManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescSet(this Self& self) noexcept {
        return std::forward_like<Self>(self).descSets_[0];
    }

    inline void updateDescSets(const ImageManager& srcImageMgr, const ImageManager& dstImageMgr);

private:
    const DeviceManager& deviceMgr_;      // FIXME: UAF
    const DescPoolManager& descPoolMgr_;  // FIXME: UAF
    std::vector<vk::DescriptorSet> descSets_;
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
    descSets_ = device.allocateDescriptorSets(descSetAllocInfo);
}

inline DescSetManager::~DescSetManager() noexcept {
    // TODO: maybe free sth. here
}

void DescSetManager::updateDescSets(const ImageManager& srcImageMgr, const ImageManager& dstImageMgr) {
    // Src
    vk::DescriptorImageInfo srcImageInfo;
    srcImageInfo.setImageView(srcImageMgr.getImageView());
    srcImageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet srcWriteDescSet;
    srcWriteDescSet.setDstSet(getDescSet());
    srcWriteDescSet.setDstBinding(0);
    srcWriteDescSet.setDescriptorCount(1);
    srcWriteDescSet.setDescriptorType(vk::DescriptorType::eStorageImage);
    srcWriteDescSet.setImageInfo(srcImageInfo);

    // Dst
    vk::DescriptorImageInfo dstImageInfo;
    dstImageInfo.setImageView(dstImageMgr.getImageView());
    dstImageInfo.setImageLayout(vk::ImageLayout::eGeneral);

    vk::WriteDescriptorSet dstWriteDescSet;
    dstWriteDescSet.setDstSet(getDescSet());
    dstWriteDescSet.setDstBinding(1);
    dstWriteDescSet.setDescriptorCount(1);
    dstWriteDescSet.setDescriptorType(vk::DescriptorType::eStorageImage);
    dstWriteDescSet.setImageInfo(dstImageInfo);

    const auto& device = deviceMgr_.getDevice();
    std::array writeDescSets{srcWriteDescSet, dstWriteDescSet};
    device.updateDescriptorSets(writeDescSets, nullptr);
}

}  // namespace vkc
