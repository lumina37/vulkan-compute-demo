#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/sampler.hpp"
#endif

namespace vkc {

SamplerManager::SamplerManager(const std::shared_ptr<DeviceManager>& pDeviceMgr) : pDeviceMgr_(pDeviceMgr) {
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.setMagFilter(vk::Filter::eLinear);
    samplerInfo.setMinFilter(vk::Filter::eLinear);

    auto& device = pDeviceMgr->getDevice();
    sampler_ = device.createSampler(samplerInfo);

    // Image Info
    samplerInfo_.setSampler(sampler_);
}

SamplerManager::~SamplerManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
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