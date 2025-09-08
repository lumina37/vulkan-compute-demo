#pragma once

#include <expected>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class SamplerBox {
    SamplerBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Sampler sampler,
               vk::DescriptorImageInfo samplerInfo) noexcept;

public:
    SamplerBox(const SamplerBox&) = delete;
    SamplerBox(SamplerBox&& rhs) noexcept;
    ~SamplerBox() noexcept;

    [[nodiscard]] static std::expected<SamplerBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox) noexcept;

    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept { return vk::DescriptorType::eSampler; }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::Sampler sampler_;
    vk::DescriptorImageInfo descSamplerInfo_;
};

constexpr vk::DescriptorSetLayoutBinding SamplerBox::draftDescSetLayoutBinding() noexcept {
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
