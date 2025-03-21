#include <cstddef>
#include <span>

#include <vulkan/vulkan.hpp>

#include "vkc/device.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/ssbo.hpp"
#endif

namespace vkc {

SSBOManager::SSBOManager(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, const vk::DeviceSize size)
    : deviceMgr_(deviceMgr), size_(size) {
    auto& device = deviceMgr.getDevice();

    // Buffer
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eStorageBuffer);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    buffer_ = device.createBuffer(bufferInfo);

    _hp::allocBufferMemory(phyDeviceMgr, deviceMgr, buffer_, vk::MemoryPropertyFlagBits::eHostVisible, memory_);
    device.bindBufferMemory(buffer_, memory_, 0);

    // Descriptor Buffer Info
    descBufferInfo_.setBuffer(buffer_);
    descBufferInfo_.setRange(size);
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
    writeDescSet.setBufferInfo(descBufferInfo_);
    return writeDescSet;
}

vk::Result SSBOManager::uploadFrom(const std::span<std::byte> data) {
    return _hp::uploadFrom(deviceMgr_, memory_, data);
}

vk::Result SSBOManager::downloadTo(std::span<std::byte> data) { return _hp::downloadTo(deviceMgr_, memory_, data); }

}  // namespace vkc
