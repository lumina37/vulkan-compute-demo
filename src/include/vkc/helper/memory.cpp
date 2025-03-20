#include <cstdint>
#include <ranges>

#include <vulkan/vulkan.hpp>

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/memory.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

uint32_t findMemoryType(const vk::PhysicalDevice& physicalDevice, const uint32_t typeFilter,
                                      const vk::MemoryPropertyFlags memProps) {
    const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (const auto& [idx, memType] : rgs::views::enumerate(memProperties.memoryTypes)) {
        if ((typeFilter & (1 << idx)) && (memType.propertyFlags & memProps) == memProps) {
            return idx;
        }
    }

    return 0;
}

void allocMemoryForBuffer(const vk::PhysicalDevice& physicalDevice, const vk::Device& device,
                                        const vk::MemoryPropertyFlags memProps, vk::Buffer& buffer,
                                        vk::DeviceMemory& bufferMemory) {
    vk::MemoryAllocateInfo allocInfo;
    const vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, memProps);
    bufferMemory = device.allocateMemory(allocInfo);

    device.bindBufferMemory(buffer, bufferMemory, 0);
}

}  // namespace vkc
