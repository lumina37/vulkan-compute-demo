#pragma once

#include <cstddef>
#include <expected>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class UniformBufferBox {
    UniformBufferBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DeviceSize size, vk::DeviceMemory memory,
                         vk::Buffer buffer, vk::DescriptorBufferInfo descBufferInfo) noexcept;

public:
    UniformBufferBox(UniformBufferBox&& rhs) noexcept;
    ~UniformBufferBox() noexcept;

    [[nodiscard]] static std::expected<UniformBufferBox, Error> create(PhyDeviceBox& phyDeviceBox,
                                                                           std::shared_ptr<DeviceBox> pDeviceBox,
                                                                           vk::DeviceSize size) noexcept;

    [[nodiscard]] vk::DeviceSize getSize() const noexcept { return size_; }
    [[nodiscard]] vk::Buffer getBuffer() const noexcept { return buffer_; }
    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eUniformBuffer;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

    [[nodiscard]] std::expected<void, Error> upload(const std::byte* pSrc) noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::DeviceSize size_;
    vk::DeviceMemory memory_;
    vk::Buffer buffer_;
    vk::DescriptorBufferInfo descBufferInfo_;
};

constexpr vk::DescriptorSetLayoutBinding UniformBufferBox::draftDescSetLayoutBinding() noexcept {
    vk::DescriptorSetLayoutBinding binding;
    binding.setDescriptorCount(1);
    binding.setDescriptorType(getDescType());
    binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return binding;
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/uniform_buffer.cpp"
#endif
