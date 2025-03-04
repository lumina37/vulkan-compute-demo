#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device.hpp"

namespace vkc {

class SSBOManager {
public:
    inline SSBOManager(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, const vk::DeviceSize size);
    inline ~SSBOManager() noexcept;

    [[nodiscard]] vk::DeviceSize getSize() const noexcept { return size_; }

    template <typename Self>
    [[nodiscard]] auto&& getMemory(this Self&& self) noexcept {
        return std::forward_like<Self>(self).memory_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getBuffer(this Self&& self) noexcept {
        return std::forward_like<Self>(self).buffer_;
    }

    [[nodiscard]] constexpr inline vk::DescriptorType getDescType() const noexcept {
        return vk::DescriptorType::eStorageBuffer;
    }
    [[nodiscard]] inline vk::WriteDescriptorSet draftWriteDescSet() const noexcept;

    inline vk::Result uploadFrom(const std::span<std::byte> data);
    inline vk::Result downloadTo(std::span<std::byte> data);

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::DeviceSize size_;
    vk::DeviceMemory memory_;
    vk::Buffer buffer_;
    vk::DescriptorBufferInfo bufferInfo_;
};

SSBOManager::SSBOManager(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, const vk::DeviceSize size)
    : deviceMgr_(deviceMgr), size_(size) {
    auto& physicalDevice = phyDeviceMgr.getPhysicalDevice();
    auto& device = deviceMgr.getDevice();

    // Buffer
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eStorageBuffer);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    buffer_ = device.createBuffer(bufferInfo);

    allocMemoryForBuffer(physicalDevice, device, vk::MemoryPropertyFlagBits::eHostVisible, buffer_, memory_);

    // Buffer Info
    bufferInfo_.setBuffer(buffer_);
    bufferInfo_.setRange(size);
}

SSBOManager::~SSBOManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyBuffer(buffer_);
    device.freeMemory(memory_);
}

vk::WriteDescriptorSet SSBOManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(getDescType());
    writeDescSet.setBufferInfo(bufferInfo_);
    return writeDescSet;
}

inline vk::Result SSBOManager::uploadFrom(const std::span<std::byte> data) {
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

inline vk::Result SSBOManager::downloadTo(std::span<std::byte> data) {
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
