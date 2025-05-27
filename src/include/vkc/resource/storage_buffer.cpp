#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/storage_buffer.hpp"
#endif

namespace vkc {

StorageBufferManager::StorageBufferManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::DeviceSize size,
                                           vk::DeviceMemory memory, vk::Buffer buffer,
                                           vk::DescriptorBufferInfo descBufferInfo) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)),
      size_(size),
      memory_(memory),
      buffer_(buffer),
      descBufferInfo_(descBufferInfo) {}

StorageBufferManager::StorageBufferManager(StorageBufferManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      size_(rhs.size_),
      memory_(std::exchange(rhs.memory_, nullptr)),
      buffer_(std::exchange(rhs.buffer_, nullptr)),
      descBufferInfo_(std::exchange(rhs.descBufferInfo_, {})) {}

StorageBufferManager::~StorageBufferManager() noexcept {
    if (pDeviceMgr_ == nullptr) return;
    vk::Device device = pDeviceMgr_->getDevice();

    if (buffer_ != nullptr) {
        device.destroyBuffer(buffer_);
        buffer_ = nullptr;
    }
    if (memory_ != nullptr) {
        device.freeMemory(memory_);
        memory_ = nullptr;
    }
}

std::expected<StorageBufferManager, Error> StorageBufferManager::create(PhyDeviceManager& phyDeviceMgr,
                                                                        std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                        vk::DeviceSize size) noexcept {
    vk::Device device = pDeviceMgr->getDevice();

    // Buffer
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eStorageBuffer);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    auto [bufferRes, buffer] = device.createBuffer(bufferInfo);
    if (bufferRes != vk::Result::eSuccess) {
        return std::unexpected{Error{bufferRes}};
    }

    vk::DeviceMemory memory;
    auto allocRes =
        _hp::allocBufferMemory(phyDeviceMgr, *pDeviceMgr, buffer, vk::MemoryPropertyFlagBits::eHostVisible, memory);
    if (!allocRes) {
        return std::unexpected{Error{std::move(allocRes.error())}};
    }

    const vk::Result bindRes = device.bindBufferMemory(buffer, memory, 0);
    if (bindRes != vk::Result::eSuccess) {
        return std::unexpected{Error{bindRes}};
    }

    // Descriptor Buffer Info
    vk::DescriptorBufferInfo descBufferInfo;
    descBufferInfo.setBuffer(buffer);
    descBufferInfo.setRange(size);

    return StorageBufferManager{std::move(pDeviceMgr), size, memory, buffer, descBufferInfo};
}

vk::WriteDescriptorSet StorageBufferManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(getDescType());
    writeDescSet.setBufferInfo(descBufferInfo_);
    return writeDescSet;
}

std::expected<void, Error> StorageBufferManager::uploadFrom(const std::span<const std::byte> data) noexcept {
    return _hp::uploadFrom(*pDeviceMgr_, memory_, data);
}

std::expected<void, Error> StorageBufferManager::downloadTo(const std::span<std::byte> data) noexcept {
    return _hp::downloadTo(*pDeviceMgr_, memory_, data);
}

}  // namespace vkc
