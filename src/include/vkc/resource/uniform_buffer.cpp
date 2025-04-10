#include <cstddef>
#include <memory>
#include <span>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/uniform_buffer.hpp"
#endif

namespace vkc {

UBOManager::UBOManager(const PhysicalDeviceManager& phyDeviceMgr, const std::shared_ptr<DeviceManager>& pDeviceMgr,
                       const vk::DeviceSize size)
    : pDeviceMgr_(pDeviceMgr), size_(size) {
    auto& device = pDeviceMgr->getDevice();

    // Buffer
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    buffer_ = device.createBuffer(bufferInfo);

    _hp::allocBufferMemory(phyDeviceMgr, *pDeviceMgr, buffer_, vk::MemoryPropertyFlagBits::eHostVisible, memory_);
    device.bindBufferMemory(buffer_, memory_, 0);

    // Descriptor Buffer Info
    descBufferInfo_.setBuffer(buffer_);
    descBufferInfo_.setRange(size);
}

UBOManager::~UBOManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    device.destroyBuffer(buffer_);
    device.freeMemory(memory_);
}

vk::WriteDescriptorSet UBOManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(getDescType());
    writeDescSet.setBufferInfo(descBufferInfo_);
    return writeDescSet;
}

vk::Result UBOManager::uploadFrom(const std::span<const std::byte> data) {
    return _hp::uploadFrom(*pDeviceMgr_, memory_, data);
}

vk::Result UBOManager::downloadTo(const std::span<std::byte> data) { return _hp::downloadTo(*pDeviceMgr_, memory_, data); }

}  // namespace vkc
