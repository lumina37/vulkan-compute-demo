#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/buffer.hpp"
#endif

namespace vkc {

BufferBox::BufferBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Buffer buffer, vk::DeviceSize size,
                     vk::BufferUsageFlags usage) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), buffer_(buffer), size_(size), usage_(usage) {}

BufferBox::BufferBox(BufferBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      buffer_(std::exchange(rhs.buffer_, nullptr)),
      size_(rhs.size_),
      usage_(rhs.usage_) {}

BufferBox::~BufferBox() noexcept {
    if (buffer_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyBuffer(buffer_);
    buffer_ = nullptr;
}

std::expected<BufferBox, Error> BufferBox::create(std::shared_ptr<DeviceBox> pDeviceBox, vk::DeviceSize size,
                                                  vk::BufferUsageFlags usage) noexcept {
    vk::Device device = pDeviceBox->getDevice();

    vk::BufferCreateInfo bufferInfo;
    bufferInfo.setSize(size);
    bufferInfo.setUsage(usage);
    bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    auto [bufferRes, buffer] = device.createBuffer(bufferInfo);
    if (bufferRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bufferRes}};
    }

    return BufferBox{std::move(pDeviceBox), buffer, size, usage};
}

vk::MemoryRequirements BufferBox::getMemoryRequirements() const noexcept {
    return _hp::getMemoryRequirements(*pDeviceBox_, buffer_);
}

std::expected<void, Error> BufferBox::bind(MemoryBox& memoryBox) noexcept {
    vk::Device device = pDeviceBox_->getDevice();

    const auto bindRes = device.bindBufferMemory(buffer_, memoryBox.getVkDeviceMemory(), 0);
    if (bindRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bindRes}};
    }

    return {};
}

}  // namespace vkc
