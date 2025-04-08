#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"

namespace vkc::_hp {

uint32_t findMemoryTypeIdx(const PhysicalDeviceManager& phyDeviceMgr, uint32_t supportedMemType,
                           vk::MemoryPropertyFlags memProps);

void allocBufferMemory(const PhysicalDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, vk::Buffer& buffer,
                       vk::MemoryPropertyFlags memProps, vk::DeviceMemory& bufferMemory);

void allocImageMemory(const PhysicalDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, vk::Image& image,
                      vk::MemoryPropertyFlags memProps, vk::DeviceMemory& bufferMemory);

vk::Result uploadFrom(DeviceManager& deviceMgr, vk::DeviceMemory& memory, std::span<const std::byte> data);

vk::Result downloadTo(DeviceManager& deviceMgr, const vk::DeviceMemory& memory, std::span<std::byte> data);

}  // namespace vkc::_hp

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/memory.cpp"
#endif
