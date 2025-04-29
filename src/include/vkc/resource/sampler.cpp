#pragma once

#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/sampler.hpp"
#endif

namespace vkc {

SamplerManager::SamplerManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Sampler sampler,
                               vk::DescriptorImageInfo samplerInfo) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), sampler_(sampler), descSamplerInfo_(samplerInfo) {}

SamplerManager::SamplerManager(SamplerManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      sampler_(std::exchange(rhs.sampler_, nullptr)),
      descSamplerInfo_(std::exchange(rhs.descSamplerInfo_, {})) {}

SamplerManager::~SamplerManager() noexcept {
    if (sampler_ == nullptr) return;
    auto& device = pDeviceMgr_->getDevice();
    device.destroySampler(sampler_);
    sampler_ = nullptr;
    descSamplerInfo_.setSampler(nullptr);
}

std::expected<SamplerManager, Error> SamplerManager::create(std::shared_ptr<DeviceManager> pDeviceMgr) noexcept {
    vk::SamplerCreateInfo samplerCreateInfo;
    samplerCreateInfo.setMagFilter(vk::Filter::eLinear);
    samplerCreateInfo.setMinFilter(vk::Filter::eLinear);
    samplerCreateInfo.setAddressModeU(vk::SamplerAddressMode::eMirroredRepeat);
    samplerCreateInfo.setAddressModeV(vk::SamplerAddressMode::eMirroredRepeat);
    samplerCreateInfo.setAddressModeW(vk::SamplerAddressMode::eMirroredRepeat);

    auto& device = pDeviceMgr->getDevice();
    const auto [samplerRes, sampler] = device.createSampler(samplerCreateInfo);
    if (samplerRes != vk::Result::eSuccess) {
        return std::unexpected{Error{samplerRes}};
    }

    // Image Info
    vk::DescriptorImageInfo samplerInfo;
    samplerInfo.setSampler(sampler);

    return SamplerManager{std::move(pDeviceMgr), sampler, samplerInfo};
}

vk::WriteDescriptorSet SamplerManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(vk::DescriptorType::eSampler);
    writeDescSet.setImageInfo(descSamplerInfo_);
    return writeDescSet;
}

}  // namespace vkc