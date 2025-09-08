#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class MemoryBox {
    MemoryBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DeviceMemory memory,
              const vk::MemoryRequirements& requirements) noexcept;

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
    [[nodiscard]] const vk::MemoryRequirements& getRequirements() const noexcept { return requirements_; }

    [[nodiscard]] std::expected<void*, Error> memMap() noexcept;
    void memUnmap() noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::DeviceMemory memory_;
    vk::MemoryRequirements requirements_;
};

namespace _hp {

[[nodiscard]] vk::MemoryRequirements getMemoryRequirements(const DeviceBox& deviceBox, vk::Image image) noexcept;

[[nodiscard]] vk::MemoryRequirements getMemoryRequirements(const DeviceBox& deviceBox, vk::Buffer buffer) noexcept;

[[nodiscard]] std::expected<uint32_t, Error> findMemoryTypeIdx(vk::PhysicalDevice physicalDevice,
                                                               uint32_t supportedMemType,
                                                               vk::MemoryPropertyFlags memProps) noexcept;

}  // namespace _hp

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/memory.cpp"
#endif
