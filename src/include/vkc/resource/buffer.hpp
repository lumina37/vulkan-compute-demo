#pragma once

#include <expected>
#include <memory>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/memory.hpp"

namespace vkc {

class BufferBox {
    BufferBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Buffer buffer, vk::DeviceSize size,
              vk::BufferUsageFlags usage) noexcept;

public:
    BufferBox(const BufferBox&) = delete;
    BufferBox(BufferBox&& rhs) noexcept;
    ~BufferBox() noexcept;

    [[nodiscard]] static std::expected<BufferBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                vk::DeviceSize size,
                                                                vk::BufferUsageFlags usage) noexcept;

    [[nodiscard]] vk::DeviceSize getSize() const noexcept { return size_; }
    [[nodiscard]] vk::Buffer getVkBuffer() noexcept { return buffer_; }
    [[nodiscard]] vk::Buffer getVkBuffer() const noexcept { return buffer_; }
    [[nodiscard]] vk::MemoryRequirements getMemoryRequirements() const noexcept;
    [[nodiscard]] std::expected<void, Error> bind(MemoryBox& memoryBox) noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::Buffer buffer_;
    vk::DeviceSize size_;
    vk::BufferUsageFlags usage_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/buffer.cpp"
#endif
