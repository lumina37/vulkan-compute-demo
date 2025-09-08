#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class MemoryBox {
    MemoryBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DeviceMemory memory) noexcept;

public:
    MemoryBox(MemoryBox&& rhs) noexcept;
    ~MemoryBox() noexcept;

    [[nodiscard]] static std::expected<MemoryBox, Error> createByIndex(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                       const vk::MemoryRequirements& requirements,
                                                                       uint32_t memTypeIndex) noexcept;
    [[nodiscard]] static std::expected<MemoryBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                const vk::MemoryRequirements& requirements,
                                                                vk::MemoryPropertyFlags props) noexcept;

    [[nodiscard]] vk::DeviceMemory getDeviceMemory() const noexcept { return memory_; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::DeviceMemory memory_;
};

namespace _hp {

[[nodiscard]] vk::MemoryRequirements getMemoryRequirements(const DeviceBox& deviceBox, vk::Image image) noexcept;

[[nodiscard]] vk::MemoryRequirements getMemoryRequirements(const DeviceBox& deviceBox, vk::Buffer buffer) noexcept;

[[nodiscard]] std::expected<uint32_t, Error> findMemoryTypeIdx(vk::PhysicalDevice physicalDevice,
                                                               uint32_t supportedMemType,
                                                               vk::MemoryPropertyFlags memProps) noexcept;

class MemMapBox {
    MemMapBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DeviceMemory memory, void* mapPtr) noexcept;

public:
    MemMapBox(MemMapBox&& rhs) noexcept;
    ~MemMapBox() noexcept;

    [[nodiscard]] static std::expected<MemMapBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                vk::DeviceMemory memory, size_t size) noexcept;

    [[nodiscard]] void* getMapPtr() const noexcept { return mapPtr_; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::DeviceMemory memory_;
    void* mapPtr_;
};

}  // namespace _hp

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/memory.cpp"
#endif
