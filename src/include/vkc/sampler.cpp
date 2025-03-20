#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/sampler.hpp"
#endif

namespace vkc {

SamplerManager::SamplerManager(DeviceManager& deviceMgr) : deviceMgr_(deviceMgr) {
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.setMagFilter(vk::Filter::eLinear);
    samplerInfo.setMinFilter(vk::Filter::eLinear);

    auto& device = deviceMgr.getDevice();
    sampler_ = device.createSampler(samplerInfo);

    // Image Info
    samplerInfo_.setSampler(sampler_);
}

SamplerManager::~SamplerManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroySampler(sampler_);
}

vk::WriteDescriptorSet SamplerManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(vk::DescriptorType::eSampler);
    writeDescSet.setImageInfo(samplerInfo_);
    return writeDescSet;
}

}  // namespace vkc