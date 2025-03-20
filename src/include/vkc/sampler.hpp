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

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::Sampler sampler_;
    vk::DescriptorImageInfo samplerInfo_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/sampler.cpp"
#endif
