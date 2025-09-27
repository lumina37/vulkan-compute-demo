#pragma once

#include <cstddef>
#include <expected>
#include <memory>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/buffer.hpp"
#include "vkc/resource/memory.hpp"

namespace vkc {

class StorageBufferBox {
    StorageBufferBox(BufferBox&& bufferBox, MemoryBox&& memoryBox,
                     const vk::DescriptorBufferInfo& descBufferInfo) noexcept;

public:
    StorageBufferBox(const StorageBufferBox&) = delete;
    StorageBufferBox(StorageBufferBox&& rhs) noexcept = default;
    ~StorageBufferBox() noexcept = default;

    [[nodiscard]] static std::expected<StorageBufferBox, Error> create(std::shared_ptr<DeviceBox>& pDeviceBox,
                                                                       vk::DeviceSize size) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getBufferBox(this Self&& self) noexcept {
        return std::forward_like<Self>(self).bufferBox_;
    }

    [[nodiscard]] vk::DeviceSize getSize() const noexcept { return bufferBox_.getSize(); }

    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eStorageBuffer;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

    [[nodiscard]] std::expected<void, Error> upload(const std::byte* pSrc) noexcept;
    [[nodiscard]] std::expected<void, Error> download(std::byte* pDst) noexcept;

private:
    BufferBox bufferBox_;
    MemoryBox memoryBox_;
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
