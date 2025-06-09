#pragma once

#include <cstddef>
#include <expected>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class StorageBufferBox {
    StorageBufferBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DeviceSize size, vk::DeviceMemory memory,
                         vk::Buffer buffer, vk::DescriptorBufferInfo descBufferInfo) noexcept;

public:
    StorageBufferBox(const StorageBufferBox&) = delete;
    StorageBufferBox(StorageBufferBox&& rhs) noexcept;
    ~StorageBufferBox() noexcept;

    [[nodiscard]] static std::expected<StorageBufferBox, Error> create(PhyDeviceBox& phyDeviceBox,
                                                                           std::shared_ptr<DeviceBox> pDeviceBox,
                                                                           vk::DeviceSize size) noexcept;

    [[nodiscard]] vk::DeviceSize getSize() const noexcept { return size_; }

    [[nodiscard]] vk::Buffer getBuffer() noexcept { return buffer_; }
    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eStorageBuffer;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

    [[nodiscard]] std::expected<void, Error> upload(const std::byte* pSrc) noexcept;
    [[nodiscard]] std::expected<void, Error> download(std::byte* pDst) noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::DeviceSize size_;
    vk::DeviceMemory memory_;
    vk::Buffer buffer_;
    vk::DescriptorBufferInfo descBufferInfo_;
};

constexpr vk::DescriptorSetLayoutBinding StorageBufferBox::draftDescSetLayoutBinding() noexcept {
    vk::DescriptorSetLayoutBinding binding;
    binding.setDescriptorCount(1);
    binding.setDescriptorType(getDescType());
    binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return binding;
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/storage_buffer.cpp"
#endif
