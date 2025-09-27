#include <cstddef>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/buffer.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/storage_buffer.hpp"
#endif

namespace vkc {

StorageBufferBox::StorageBufferBox(BufferBox&& bufferBox, MemoryBox&& memoryBox,
                                   const vk::DescriptorBufferInfo& descBufferInfo) noexcept
    : bufferBox_(std::move(bufferBox)), memoryBox_(std::move(memoryBox)), descBufferInfo_(descBufferInfo) {}

std::expected<StorageBufferBox, Error> StorageBufferBox::create(std::shared_ptr<DeviceBox>& pDeviceBox,
                                                                vk::DeviceSize size) noexcept {
    auto bufferBoxRes = BufferBox::create(pDeviceBox, size, vk::BufferUsageFlagBits::eStorageBuffer);
    if (!bufferBoxRes) return std::unexpected{std::move(bufferBoxRes.error())};
    BufferBox& bufferBox = bufferBoxRes.value();

    auto memoryBoxRes =
        MemoryBox::create(pDeviceBox, bufferBox.getMemoryRequirements(), vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!memoryBoxRes) return std::unexpected{std::move(memoryBoxRes.error())};
    MemoryBox& memoryBox = memoryBoxRes.value();

    const auto bindRes = bufferBox.bind(memoryBox);
    if (!bindRes) return std::unexpected{std::move(bindRes.error())};

    // Descriptor Buffer Info
    vk::DescriptorBufferInfo descBufferInfo;
    descBufferInfo.setBuffer(bufferBox.getVkBuffer());
    descBufferInfo.setRange(size);

    return StorageBufferBox{std::move(bufferBox), std::move(memoryBox), descBufferInfo};
}

vk::WriteDescriptorSet StorageBufferBox::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(getDescType());
    writeDescSet.setBufferInfo(descBufferInfo_);
    return writeDescSet;
}

std::expected<void, Error> StorageBufferBox::upload(const std::byte* pSrc) noexcept {
    auto mmapRes = memoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    std::memcpy(mapPtr, pSrc, getSize());

    memoryBox_.memUnmap();

    return {};
}

std::expected<void, Error> StorageBufferBox::download(std::byte* pDst) noexcept {
    auto mmapRes = memoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    std::memcpy(pDst, mapPtr, getSize());

    memoryBox_.memUnmap();

    return {};
}

}  // namespace vkc
