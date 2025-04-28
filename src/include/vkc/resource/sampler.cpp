#pragma once

#include <expected>
#include <memory>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/sampler.hpp"
#endif

namespace vkc {

SamplerManager::SamplerManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Sampler sampler,
                               vk::DescriptorImageInfo samplerInfo) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), sampler_(sampler), samplerInfo_(samplerInfo) {}

SamplerManager::SamplerManager(SamplerManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      sampler_(std::exchange(rhs.sampler_, nullptr)),
      samplerInfo_(std::exchange(rhs.samplerInfo_, {})) {}

SamplerManager::~SamplerManager() noexcept {
    if (sampler_ == nullptr) return;
    auto& device = pDeviceMgr_->getDevice();
    device.destroySampler(sampler_);
    sampler_ = nullptr;
    samplerInfo_ = vk::DescriptorImageInfo{};
}

std::expected<SamplerManager, Error> SamplerManager::create(std::shared_ptr<DeviceManager> pDeviceMgr) noexcept {
    vk::SamplerCreateInfo samplerCreateInfo;
    samplerCreateInfo.setMagFilter(vk::Filter::eLinear);
    samplerCreateInfo.setMinFilter(vk::Filter::eLinear);
    samplerCreateInfo.setAddressModeU(vk::SamplerAddressMode::eMirroredRepeat);
    samplerCreateInfo.setAddressModeV(vk::SamplerAddressMode::eMirroredRepeat);
    samplerCreateInfo.setAddressModeW(vk::SamplerAddressMode::eMirroredRepeat);

    auto& device = pDeviceMgr->getDevice();
    const vk::Sampler sampler = device.createSampler(samplerCreateInfo);

    // Image Info
    vk::DescriptorImageInfo samplerInfo;
    samplerInfo.setSampler(sampler);

    return SamplerManager{std::move(pDeviceMgr), sampler, samplerInfo};
}

vk::WriteDescriptorSet SamplerManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(vk::DescriptorType::eSampler);
    writeDescSet.setImageInfo(samplerInfo_);
    return writeDescSet;
}

}  // namespace vkc