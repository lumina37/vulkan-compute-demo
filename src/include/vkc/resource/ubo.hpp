#pragma once

#include <cstddef>
#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"

namespace vkc {

class UBOManager {
public:
    UBOManager(const PhysicalDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, vk::DeviceSize size);
    ~UBOManager() noexcept;

    [[nodiscard]] vk::DeviceSize getSize() const noexcept { return size_; }

    template <typename Self>
    [[nodiscard]] auto&& getMemory(this Self&& self) noexcept {
        return std::forward_like<Self>(self).memory_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getBuffer(this Self&& self) noexcept {
        return std::forward_like<Self>(self).buffer_;
    }

    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eUniformBuffer;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

    vk::Result uploadFrom(std::span<std::byte> data);
    vk::Result downloadTo(std::span<std::byte> data);

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::DeviceSize size_;
    vk::DeviceMemory memory_;
    vk::Buffer buffer_;
    vk::DescriptorBufferInfo descBufferInfo_;
};

constexpr vk::DescriptorSetLayoutBinding UBOManager::draftDescSetLayoutBinding() noexcept {
    vk::DescriptorSetLayoutBinding binding;
    binding.setDescriptorCount(1);
    binding.setDescriptorType(getDescType());
    binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return binding;
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/ubo.cpp"
#endif
