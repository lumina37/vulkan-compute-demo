#include <cstddef>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/storage_buffer.hpp"
#endif

namespace vkc {

StorageBufferBox::StorageBufferBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DeviceSize size, MemoryBox&& memoryBox,
                                   vk::Buffer buffer, vk::DescriptorBufferInfo descBufferInfo) noexcept
    : pDeviceBox_(std::move(pDeviceBox)),
      size_(size),
      memoryBox_(std::move(memoryBox)),
      buffer_(buffer),
      descBufferInfo_(descBufferInfo) {}

StorageBufferBox::StorageBufferBox(StorageBufferBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      size_(rhs.size_),
      memoryBox_(std::move(rhs.memoryBox_)),
      buffer_(std::exchange(rhs.buffer_, nullptr)),
      descBufferInfo_(std::exchange(rhs.descBufferInfo_, {})) {}

StorageBufferBox::~StorageBufferBox() noexcept {
    if (buffer_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyBuffer(buffer_);
    buffer_ = nullptr;
}

std::expected<StorageBufferBox, Error> StorageBufferBox::create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                vk::DeviceSize size) noexcept {
    vk::Device device = pDeviceBox->getDevice();

    // Buffer
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eStorageBuffer);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    auto [bufferRes, buffer] = device.createBuffer(bufferInfo);
    if (bufferRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bufferRes}};
    }

    const vk::MemoryRequirements stagingMemoryReq = _hp::getMemoryRequirements(*pDeviceBox, buffer);
    auto memoryBoxRes = MemoryBox::create(pDeviceBox, stagingMemoryReq, vk::MemoryPropertyFlagBits::eHostVisible);
    if (!memoryBoxRes) return std::unexpected{std::move(memoryBoxRes.error())};
    MemoryBox memoryBox = std::move(memoryBoxRes.value());

    const auto bindRes = device.bindBufferMemory(buffer, memoryBox.getDeviceMemory(), 0);
    if (bindRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bindRes}};
    }

    // Descriptor Buffer Info
    vk::DescriptorBufferInfo descBufferInfo;
    descBufferInfo.setBuffer(buffer);
    descBufferInfo.setRange(size);

    return StorageBufferBox{std::move(pDeviceBox), size, std::move(memoryBox), buffer, descBufferInfo};
}

vk::WriteDescriptorSet StorageBufferBox::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(getDescType());
    writeDescSet.setBufferInfo(descBufferInfo_);
    return writeDescSet;
}

std::expected<void, Error> StorageBufferBox::upload(const std::byte* pSrc) noexcept {
    auto mmapRes = _hp::MemMapBox::create(pDeviceBox_, memoryBox_.getDeviceMemory(), size_);
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapBox = mmapRes.value();

    std::memcpy(mmapBox.getMapPtr(), pSrc, size_);

    return {};
}

std::expected<void, Error> StorageBufferBox::download(std::byte* pDst) noexcept {
    auto mmapRes = _hp::MemMapBox::create(pDeviceBox_, memoryBox_.getDeviceMemory(), size_);
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapBox = mmapRes.value();

    std::memcpy(pDst, mmapBox.getMapPtr(), size_);

    return {};
}

}  // namespace vkc
