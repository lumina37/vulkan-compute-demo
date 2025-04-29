#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc::_hp {

std::expected<uint32_t, Error> findMemoryTypeIdx(const PhysicalDeviceManager& phyDeviceMgr, uint32_t supportedMemType,
                                                 vk::MemoryPropertyFlags memProps) noexcept;

std::expected<void, Error> allocBufferMemory(const PhysicalDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr,
                                             vk::Buffer& buffer, vk::MemoryPropertyFlags memProps,
                                             vk::DeviceMemory& bufferMemory) noexcept;

std::expected<void, Error> allocImageMemory(const PhysicalDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr,
                                            vk::Image& image, vk::MemoryPropertyFlags memProps,
                                            vk::DeviceMemory& bufferMemory) noexcept;

std::expected<void, Error> uploadFrom(DeviceManager& deviceMgr, vk::DeviceMemory& memory,
                                      std::span<const std::byte> data) noexcept;

std::expected<void, Error> downloadTo(DeviceManager& deviceMgr, const vk::DeviceMemory& memory,
                                      std::span<std::byte> data) noexcept;

}  // namespace vkc::_hp

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/memory.cpp"
#endif
