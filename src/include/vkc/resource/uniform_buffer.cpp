#include <memory>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/buffer.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/uniform_buffer.hpp"
#endif

namespace vkc {

UniformBufferBox::UniformBufferBox(BufferBox&& bufferBox, MemoryBox&& memoryBox,
                                   const vk::DescriptorBufferInfo& descBufferInfo) noexcept
    : bufferBox_(std::move(bufferBox)),
      memoryBox_(std::move(memoryBox)),
      descBufferInfo_(descBufferInfo),
      accessMask_(vk::AccessFlagBits::eNone) {}

std::expected<UniformBufferBox, Error> UniformBufferBox::create(std::shared_ptr<DeviceBox>& pDeviceBox,
                                                                vk::DeviceSize size) noexcept {
    constexpr vk::BufferUsageFlags bufferUsage =
        vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst;

    auto bufferBoxRes = BufferBox::create(pDeviceBox, size, bufferUsage);
    if (!bufferBoxRes) return std::unexpected{std::move(bufferBoxRes.error())};
    BufferBox& bufferBox = bufferBoxRes.value();

    auto memoryBoxRes =
        MemoryBox::create(pDeviceBox, bufferBox.getMemoryRequirements(), vk::MemoryPropertyFlagBits::eDeviceLocal);
    if (!memoryBoxRes) return std::unexpected{std::move(memoryBoxRes.error())};
    MemoryBox& memoryBox = memoryBoxRes.value();

    auto bindRes = bufferBox.bind(memoryBox);
    if (!bindRes) return std::unexpected{std::move(bindRes.error())};

    // Descriptor Buffer Info
    vk::DescriptorBufferInfo descBufferInfo;
    descBufferInfo.setBuffer(bufferBox.getVkBuffer());
    descBufferInfo.setRange(size);

    return UniformBufferBox{std::move(bufferBox), std::move(memoryBox), descBufferInfo};
}

vk::WriteDescriptorSet UniformBufferBox::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(getDescType());
    writeDescSet.setBufferInfo(descBufferInfo_);
    return writeDescSet;
}

std::expected<void, Error> UniformBufferBox::upload(const std::byte* pSrc) noexcept {
    auto mmapRes = memoryBox_.memMap();
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    void* mapPtr = mmapRes.value();

    std::memcpy(mapPtr, pSrc, getSize());

    memoryBox_.memUnmap();

    return {};
}

}  // namespace vkc
