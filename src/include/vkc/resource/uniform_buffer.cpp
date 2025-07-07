#include <cstddef>
#include <expected>
#include <memory>
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

UniformBufferBox::UniformBufferBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DeviceSize size,
                                   vk::DeviceMemory memory, vk::Buffer buffer,
                                   vk::DescriptorBufferInfo descBufferInfo) noexcept
    : pDeviceBox_(std::move(pDeviceBox)),
      size_(size),
      memory_(memory),
      buffer_(buffer),
      descBufferInfo_(descBufferInfo) {}

UniformBufferBox::UniformBufferBox(UniformBufferBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      size_(rhs.size_),
      memory_(std::exchange(rhs.memory_, nullptr)),
      buffer_(std::exchange(rhs.buffer_, nullptr)),
      descBufferInfo_(std::exchange(rhs.descBufferInfo_, {})) {}

UniformBufferBox::~UniformBufferBox() noexcept {
    if (pDeviceBox_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();

    if (buffer_ != nullptr) {
        device.destroyBuffer(buffer_);
        buffer_ = nullptr;
    }
    if (memory_ != nullptr) {
        device.freeMemory(memory_);
        memory_ = nullptr;
    }
}

std::expected<UniformBufferBox, Error> UniformBufferBox::create(PhyDeviceBox& phyDeviceBox,
                                                                std::shared_ptr<DeviceBox> pDeviceBox,
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

    vk::DeviceMemory memory;
    auto allocRes =
        _hp::allocBufferMemory(phyDeviceBox, *pDeviceBox, buffer, vk::MemoryPropertyFlagBits::eHostVisible, memory);
    if (!allocRes) {
        return std::unexpected{std::move(allocRes.error())};
    }

    const auto bindRes = device.bindBufferMemory(buffer, memory, 0);
    if (bindRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bindRes}};
    }

    // Descriptor Buffer Info
    vk::DescriptorBufferInfo descBufferInfo;
    descBufferInfo.setBuffer(buffer);
    descBufferInfo.setRange(size);

    return UniformBufferBox{std::move(pDeviceBox), size, memory, buffer, descBufferInfo};
}

vk::WriteDescriptorSet UniformBufferBox::draftWriteDescSet() const noexcept {
    vk::WriteDescriptorSet writeDescSet;
    writeDescSet.setDescriptorCount(1);
    writeDescSet.setDescriptorType(getDescType());
    writeDescSet.setBufferInfo(descBufferInfo_);
    return writeDescSet;
}

std::expected<void, Error> UniformBufferBox::upload(const std::byte* pSrc) noexcept {
    auto mmapRes = _hp::MemMapBox::create(pDeviceBox_, memory_, size_);
    if (!mmapRes) return std::unexpected{std::move(mmapRes.error())};
    auto& mmapBox = mmapRes.value();

    std::memcpy(mmapBox.getMapPtr(), pSrc, size_);

    return {};
}

}  // namespace vkc
