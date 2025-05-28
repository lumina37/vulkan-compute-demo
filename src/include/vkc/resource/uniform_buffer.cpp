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
#    include "vkc/resource/uniform_buffer.hpp"
#endif

namespace vkc {

UniformBufferManager::UniformBufferManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::DeviceSize size,
                                           vk::DeviceMemory memory, vk::Buffer buffer,
                                           vk::DescriptorBufferInfo descBufferInfo) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)),
      size_(size),
      memory_(memory),
      buffer_(buffer),
      descBufferInfo_(descBufferInfo) {}

UniformBufferManager::UniformBufferManager(UniformBufferManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      size_(rhs.size_),
      memory_(std::exchange(rhs.memory_, nullptr)),
      buffer_(std::exchange(rhs.buffer_, nullptr)),
      descBufferInfo_(std::exchange(rhs.descBufferInfo_, {})) {}

UniformBufferManager::~UniformBufferManager() noexcept {
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

std::expected<UniformBufferManager, Error> UniformBufferManager::create(PhyDeviceManager& phyDeviceMgr,
                                                                        std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                        vk::DeviceSize size) noexcept {
    vk::Device device = pDeviceMgr->getDevice();

    // Buffer
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
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

    return UniformBufferManager{std::move(pDeviceMgr), size, memory, buffer, descBufferInfo};
}

vk::WriteDescriptorSet UniformBufferManager::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(getDescType());
    writeDescSet.setBufferInfo(descBufferInfo_);
    return writeDescSet;
}

std::expected<void, Error> UniformBufferManager::uploadFrom(const std::byte* pData) noexcept {
    auto mmapRes = _hp::MemMapManager::create(pDeviceMgr_, memory_, size_);
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapMgr = mmapRes.value();

    std::memcpy(mmapMgr.getMapPtr(), pData, size_);

    return {};
}

}  // namespace vkc
