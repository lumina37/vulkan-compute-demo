#pragma once

#include <cstdint>

#include <vulkan/vulkan.hpp>

namespace vkc {

uint32_t findMemoryType(const vk::PhysicalDevice& physicalDevice, uint32_t typeFilter,
                        vk::MemoryPropertyFlags memProps);

void allocMemoryForBuffer(const vk::PhysicalDevice& physicalDevice, const vk::Device& device,
                          vk::MemoryPropertyFlags memProps, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory);

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/memory.cpp"
#endif
