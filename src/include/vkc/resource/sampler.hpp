#pragma once

#include <expected>
#include <memory>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

class SamplerManager {
    SamplerManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Sampler sampler,
                   vk::DescriptorImageInfo samplerInfo) noexcept;

public:
    SamplerManager(SamplerManager&& rhs) noexcept;
    ~SamplerManager() noexcept;

    [[nodiscard]] static std::expected<SamplerManager, Error> create(
        std::shared_ptr<DeviceManager> pDeviceMgr) noexcept;

    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept { return vk::DescriptorType::eSampler; }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::Sampler sampler_;
    vk::DescriptorImageInfo descSamplerInfo_;
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
