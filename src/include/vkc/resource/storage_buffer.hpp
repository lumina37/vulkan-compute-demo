#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <utility>

#include "vkc/helper/vulkan.hpp"

#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

class StorageBufferManager {
    StorageBufferManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::DeviceSize size, vk::DeviceMemory memory,
                         vk::Buffer buffer, vk::DescriptorBufferInfo descBufferInfo) noexcept;

public:
    StorageBufferManager(StorageBufferManager&& rhs) noexcept;
    ~StorageBufferManager() noexcept;

    [[nodiscard]] static std::expected<StorageBufferManager, Error> create(PhysicalDeviceManager& phyDeviceMgr,
                                                                           std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                           vk::DeviceSize size) noexcept;

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
        return vk::DescriptorType::eStorageBuffer;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

    vk::Result uploadFrom(std::span<const std::byte> data);
    vk::Result downloadTo(std::span<std::byte> data);

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::DeviceSize size_;
    vk::DeviceMemory memory_;
    vk::Buffer buffer_;
    vk::DescriptorBufferInfo descBufferInfo_;
};

constexpr vk::DescriptorSetLayoutBinding StorageBufferManager::draftDescSetLayoutBinding() noexcept {
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
