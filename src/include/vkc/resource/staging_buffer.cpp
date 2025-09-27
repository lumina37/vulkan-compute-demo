#include <cstddef>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/staging_buffer.hpp"
#endif

namespace vkc {

StagingBufferBox::StagingBufferBox(BufferBox&& bufferBox, MemoryBox&& memoryBox) noexcept
    : bufferBox_(std::move(bufferBox)), memoryBox_(std::move(memoryBox)) {}

std::expected<StagingBufferBox, Error> StagingBufferBox::create(std::shared_ptr<DeviceBox>& pDeviceBox,
                                                                vk::DeviceSize size, StorageType bufferType) noexcept {
    vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eTransferSrc;

    if (StorageType::ReadWrite == bufferType) {
        bufferUsage |= vk::BufferUsageFlagBits::eTransferDst;
    }

    auto bufferBoxRes = BufferBox::create(pDeviceBox, size, bufferUsage);
    if (!bufferBoxRes) return std::unexpected{std::move(bufferBoxRes.error())};
    BufferBox& bufferBox = bufferBoxRes.value();

    auto memoryBoxRes =
        MemoryBox::create(pDeviceBox, bufferBox.getMemoryRequirements(),
                          vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!memoryBoxRes) return std::unexpected{std::move(memoryBoxRes.error())};
    MemoryBox& memoryBox = memoryBoxRes.value();

    const auto bindRes = bufferBox.bind(memoryBox);
    if (!bindRes) return std::unexpected{std::move(bindRes.error())};

    return StagingBufferBox{std::move(bufferBox), std::move(memoryBox)};
}

std::expected<void, Error> StagingBufferBox::upload(const std::byte* pSrc) noexcept {
    auto mmapRes = memoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    std::memcpy(mapPtr, pSrc, getSize());

    memoryBox_.memUnmap();

    return {};
}

std::expected<void, Error> StagingBufferBox::uploadWithRoi(const std::byte* pSrc, const Extent& extent, const Roi& roi,
                                                           size_t bufferOffset, size_t bufferRowPitch) noexcept {
    auto mmapRes = memoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    size_t srcOffset = 0;
    size_t dstOffset = bufferOffset;
    for (int row = 0; row < (int)roi.extent().height; row++) {
        const std::byte* srcCursor = pSrc + srcOffset;
        std::byte* dstCursor = (std::byte*)mapPtr + dstOffset;
        std::memcpy(dstCursor, srcCursor, roi.extent().width * extent.bpp());
        srcOffset += bufferRowPitch;
        dstOffset += extent.rowPitch();
    }

    memoryBox_.memUnmap();

    return {};
}

std::expected<void, Error> StagingBufferBox::download(std::byte* pDst) noexcept {
    auto mmapRes = memoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    std::memcpy(pDst, mapPtr, getSize());

    memoryBox_.memUnmap();

    return {};
}

std::expected<void, Error> StagingBufferBox::downloadWithRoi(std::byte* pDst, const Extent& extent, const Roi& roi,
                                                             size_t bufferOffset, size_t bufferRowPitch) noexcept {
    auto mmapRes = memoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    size_t srcOffset = bufferOffset;
    size_t dstOffset = 0;
    for (int row = 0; row < (int)roi.extent().height; row++) {
        const std::byte* srcCursor = (std::byte*)mapPtr + srcOffset;
        std::byte* dstCursor = pDst + dstOffset;
        std::memcpy(dstCursor, srcCursor, roi.extent().width * extent.bpp());
        srcOffset += extent.rowPitch();
        dstOffset += bufferRowPitch;
    }

    memoryBox_.memUnmap();

    return {};
}

}  // namespace vkc
