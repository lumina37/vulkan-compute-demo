#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

namespace vkc {

class SamplerManager {
public:
    SamplerManager(DeviceManager& deviceMgr);
    ~SamplerManager() noexcept;

    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept { return vk::DescriptorType::eSampler; }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::Sampler sampler_;
    vk::DescriptorImageInfo samplerInfo_;
};

constexpr vk::DescriptorSetLayoutBinding SamplerManager::draftDescSetLayoutBinding() noexcept {
    vk::DescriptorSetLayoutBinding binding;
    binding.setDescriptorCount(1);
    binding.setDescriptorType(getDescType());
    binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return binding;
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/sampler.cpp"
#endif
