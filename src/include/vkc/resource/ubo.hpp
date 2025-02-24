#pragma once

#include <vulkan/vulkan.hpp>

namespace vkc {

class UboManager {
public:
    inline UboManager(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, const vk::DeviceSize size,
                      const vk::BufferUsageFlags usage, const vk::DescriptorType descType);
    inline ~UboManager() noexcept;

    [[nodiscard]] vk::DeviceSize getSize() const noexcept { return size_; }

    template <typename Self>
    [[nodiscard]] auto&& getMemory(this Self&& self) noexcept {
        return std::forward_like<Self>(self).memory_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getBuffer(this Self&& self) noexcept {
        return std::forward_like<Self>(self).buffer_;
    }

    [[nodiscard]] inline vk::DescriptorType getDescType() const noexcept { return descType_; }
    [[nodiscard]] inline vk::WriteDescriptorSet draftWriteDescSet() const noexcept;

    inline vk::Result uploadFrom(const std::span<std::byte> data);
    inline vk::Result downloadTo(std::span<std::byte> data);

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::DeviceSize size_;
    vk::DescriptorType descType_;
    vk::DeviceMemory memory_;
    vk::Buffer buffer_;
    vk::DescriptorBufferInfo bufferInfo_;
};

UboManager::UboManager(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, const vk::DeviceSize size,
                       const vk::BufferUsageFlags usage, const vk::DescriptorType descType)
    : deviceMgr_(deviceMgr), size_(size), descType_(descType) {
    auto& physicalDevice = phyDeviceMgr.getPhysicalDevice();
    auto& device = deviceMgr.getDevice();

    // Buffer
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(usage);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    buffer_ = device.createBuffer(bufferInfo);

    allocMemoryForBuffer(physicalDevice, device,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, buffer_,
                         memory_);

    // Buffer Info
    bufferInfo_.setBuffer(buffer_);
    bufferInfo_.setRange(size);
}

UboManager::~UboManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyBuffer(buffer_);
    device.freeMemory(memory_);
}

vk::WriteDescriptorSet UboManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(vk::DescriptorType::eUniformBuffer);
    writeDescSet.setBufferInfo(bufferInfo_);
    return writeDescSet;
}

inline vk::Result UboManager::uploadFrom(const std::span<std::byte> data) {
    auto& device = deviceMgr_.getDevice();

    // Upload to Buffer
    void* mapPtr;
    auto uploadMapResult = device.mapMemory(memory_, 0, data.size(), (vk::MemoryMapFlags)0, &mapPtr);
    if (uploadMapResult != vk::Result::eSuccess) {
        return uploadMapResult;
    }
    std::memcpy(mapPtr, data.data(), data.size());
    device.unmapMemory(memory_);

    return vk::Result::eSuccess;
}

inline vk::Result UboManager::downloadTo(std::span<std::byte> data) {
    auto& device = deviceMgr_.getDevice();

    // Download from Buffer
    void* mapPtr;
    auto downloadMapResult = device.mapMemory(memory_, 0, data.size(), (vk::MemoryMapFlags)0, &mapPtr);
    if (downloadMapResult != vk::Result::eSuccess) {
        return downloadMapResult;
    }
    std::memcpy(data.data(), mapPtr, data.size());
    device.unmapMemory(memory_);

    return vk::Result::eSuccess;
}

}  // namespace vkc
