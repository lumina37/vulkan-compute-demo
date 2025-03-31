#pragma once

#include <memory>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

namespace vkc {

class SamplerManager {
public:
    SamplerManager(const std::shared_ptr<DeviceManager>& pDeviceMgr);
    ~SamplerManager() noexcept;

    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept { return vk::DescriptorType::eSampler; }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

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
