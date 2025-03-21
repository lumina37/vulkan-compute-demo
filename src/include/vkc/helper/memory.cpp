#include <cstdint>
#include <ranges>

#include <vulkan/vulkan.hpp>

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/helper/memory.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

uint32_t findMemoryTypeIdx(const vk::PhysicalDevice& physicalDevice, const uint32_t supportedMemType,
                        const vk::MemoryPropertyFlags memProps) {
    const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (const auto [idx, memType] : rgs::views::enumerate(memProperties.memoryTypes)) {
        const bool isSupported = supportedMemType & (1 << idx);
        const bool isSufficient = (memType.propertyFlags & memProps) == memProps;
        if (isSupported && isSufficient) {
            return idx;
        }
    }

    return 0;
}

void allocMemoryForBuffer(const vk::PhysicalDevice& physicalDevice, const vk::Device& device,
                          const vk::MemoryPropertyFlags memProps, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
    vk::MemoryAllocateInfo allocInfo;
    const vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.memoryTypeIndex = findMemoryTypeIdx(physicalDevice, memRequirements.memoryTypeBits, memProps);
    bufferMemory = device.allocateMemory(allocInfo);

    device.bindBufferMemory(buffer, bufferMemory, 0);
}

}  // namespace vkc
