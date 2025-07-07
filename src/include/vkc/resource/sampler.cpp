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

SamplerBox::SamplerBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Sampler sampler,
                               vk::DescriptorImageInfo samplerInfo) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), sampler_(sampler), descSamplerInfo_(samplerInfo) {}

SamplerBox::SamplerBox(SamplerBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      sampler_(std::exchange(rhs.sampler_, nullptr)),
      descSamplerInfo_(std::exchange(rhs.descSamplerInfo_, {})) {}

SamplerBox::~SamplerBox() noexcept {
    if (sampler_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroySampler(sampler_);
    sampler_ = nullptr;
    descSamplerInfo_.setSampler(nullptr);
}

std::expected<SamplerBox, Error> SamplerBox::create(std::shared_ptr<DeviceBox> pDeviceBox) noexcept {
    vk::SamplerCreateInfo samplerCreateInfo;
    samplerCreateInfo.setMagFilter(vk::Filter::eLinear);
    samplerCreateInfo.setMinFilter(vk::Filter::eLinear);
    samplerCreateInfo.setAddressModeU(vk::SamplerAddressMode::eMirroredRepeat);
    samplerCreateInfo.setAddressModeV(vk::SamplerAddressMode::eMirroredRepeat);
    samplerCreateInfo.setAddressModeW(vk::SamplerAddressMode::eMirroredRepeat);

    vk::Device device = pDeviceBox->getDevice();
    const auto [samplerRes, sampler] = device.createSampler(samplerCreateInfo);
    if (samplerRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, samplerRes}};
    }

    // Image Info
    vk::DescriptorImageInfo samplerInfo;
    samplerInfo.setSampler(sampler);

    return SamplerBox{std::move(pDeviceBox), sampler, samplerInfo};
}

vk::WriteDescriptorSet SamplerBox::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(vk::DescriptorType::eSampler);
    writeDescSet.setImageInfo(descSamplerInfo_);
    return writeDescSet;
}

}  // namespace vkc