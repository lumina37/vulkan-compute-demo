#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

namespace vkc {

class SamplerManager {
public:
    inline SamplerManager(const DeviceManager& deviceMgr);
    inline ~SamplerManager() noexcept;

    [[nodiscard]] static inline constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eSampler;
    }
    [[nodiscard]] inline vk::WriteDescriptorSet draftWriteDescSet() const noexcept;

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::Sampler sampler_;
    vk::DescriptorImageInfo samplerInfo_;
};

SamplerManager::SamplerManager(const DeviceManager& deviceMgr) : deviceMgr_(deviceMgr) {
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.setMagFilter(vk::Filter::eLinear);
    samplerInfo.setMinFilter(vk::Filter::eLinear);

    const auto& device = deviceMgr.getDevice();
    sampler_ = device.createSampler(samplerInfo);

    // Image Info
    samplerInfo_.setSampler(sampler_);
}

inline SamplerManager::~SamplerManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
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