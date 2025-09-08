#include <cstddef>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/memory.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/uniform_buffer.hpp"
#endif

namespace vkc {

UniformBufferBox::UniformBufferBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Buffer buffer, MemoryBox&& memoryBox,
                                   const vk::DescriptorBufferInfo& descBufferInfo) noexcept
    : pDeviceBox_(std::move(pDeviceBox)),
      buffer_(buffer),
      memoryBox_(std::move(memoryBox)),
      descBufferInfo_(descBufferInfo) {}

UniformBufferBox::UniformBufferBox(UniformBufferBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      buffer_(std::exchange(rhs.buffer_, nullptr)),
      memoryBox_(std::move(rhs.memoryBox_)),
      descBufferInfo_(std::exchange(rhs.descBufferInfo_, {})) {}

UniformBufferBox::~UniformBufferBox() noexcept {
    if (buffer_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyBuffer(buffer_);
    buffer_ = nullptr;
}

std::expected<UniformBufferBox, Error> UniformBufferBox::create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                vk::DeviceSize size) noexcept {
    vk::Device device = pDeviceBox->getDevice();

    // Buffer
    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    auto [bufferRes, buffer] = device.createBuffer(bufferInfo);
    if (bufferRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bufferRes}};
    }

    const vk::MemoryRequirements stagingMemoryReq = _hp::getMemoryRequirements(*pDeviceBox, buffer);
    auto memoryBoxRes = MemoryBox::create(pDeviceBox, stagingMemoryReq, vk::MemoryPropertyFlagBits::eHostVisible);
    if (!memoryBoxRes) return std::unexpected{std::move(memoryBoxRes.error())};
    MemoryBox& memoryBox = memoryBoxRes.value();

    const auto bindRes = device.bindBufferMemory(buffer, memoryBox.getDeviceMemory(), 0);
    if (bindRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bindRes}};
    }

    // Descriptor Buffer Info
    vk::DescriptorBufferInfo descBufferInfo;
    descBufferInfo.setBuffer(buffer);
    descBufferInfo.setRange(size);

    return UniformBufferBox{std::move(pDeviceBox), buffer, std::move(memoryBox), descBufferInfo};
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
