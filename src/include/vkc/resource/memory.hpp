#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc::_hp {

[[nodiscard]] std::expected<uint32_t, Error> findMemoryTypeIdx(const PhyDeviceBox& phyDeviceBox,
                                                               uint32_t supportedMemType,
                                                               vk::MemoryPropertyFlags memProps) noexcept;

[[nodiscard]] std::expected<void, Error> allocBufferMemory(const PhyDeviceBox& phyDeviceBox, DeviceBox& deviceBox,
                                                           vk::Buffer& buffer, vk::MemoryPropertyFlags memProps,
                                                           vk::DeviceMemory& bufferMemory) noexcept;

[[nodiscard]] std::expected<void, Error> allocImageMemory(const PhyDeviceBox& phyDeviceBox, DeviceBox& deviceBox,
                                                          vk::Image& image, vk::MemoryPropertyFlags memProps,
                                                          vk::DeviceMemory& bufferMemory) noexcept;

class MemMapBox {
    MemMapBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DeviceMemory memory, void* mapPtr) noexcept;

public:
    MemMapBox(MemMapBox&& rhs) noexcept;
    ~MemMapBox() noexcept;

    [[nodiscard]] static std::expected<MemMapBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                vk::DeviceMemory& memory, size_t size) noexcept;

    [[nodiscard]] void* getMapPtr() const noexcept { return mapPtr_; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::DeviceMemory memory_;
    void* mapPtr_;
};

}  // namespace vkc::_hp

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/memory.cpp"
#endif
