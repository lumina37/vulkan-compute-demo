#include <cstdint>
#include <ranges>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/memory.hpp"
#endif

namespace vkc::_hp {

namespace rgs = std::ranges;

uint32_t findMemoryTypeIdx(const PhysicalDeviceManager& phyDeviceMgr, const uint32_t supportedMemType,
                           const vk::MemoryPropertyFlags memProps) {
    const auto& physicalDevice = phyDeviceMgr.getPhysicalDevice();
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

void allocBufferMemory(const PhysicalDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, vk::Buffer& buffer,
                       vk::MemoryPropertyFlags memProps, vk::DeviceMemory& bufferMemory) {
    auto& device = deviceMgr.getDevice();

    vk::MemoryAllocateInfo allocInfo;
    const vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.memoryTypeIndex = findMemoryTypeIdx(phyDeviceMgr, memRequirements.memoryTypeBits, memProps);
    bufferMemory = device.allocateMemory(allocInfo);
}

void allocImageMemory(const PhysicalDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, vk::Image& image,
                      vk::MemoryPropertyFlags memProps, vk::DeviceMemory& bufferMemory) {
    auto& device = deviceMgr.getDevice();

    vk::MemoryAllocateInfo allocInfo;
    const vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);
    allocInfo.setAllocationSize(memRequirements.size);
    allocInfo.memoryTypeIndex = findMemoryTypeIdx(phyDeviceMgr, memRequirements.memoryTypeBits, memProps);
    bufferMemory = device.allocateMemory(allocInfo);
}

vk::Result uploadFrom(DeviceManager& deviceMgr, vk::DeviceMemory& memory, const std::span<std::byte> data) {
    auto& device = deviceMgr.getDevice();

    // Upload to Buffer
    void* mapPtr;
    auto uploadMapResult = device.mapMemory(memory, 0, data.size(), (vk::MemoryMapFlags)0, &mapPtr);
    if (uploadMapResult != vk::Result::eSuccess) {
        return uploadMapResult;
    }
    std::memcpy(mapPtr, data.data(), data.size());
    device.unmapMemory(memory);

    return vk::Result::eSuccess;
}

vk::Result downloadTo(DeviceManager& deviceMgr, vk::DeviceMemory& memory, std::span<std::byte> data) {
    auto& device = deviceMgr.getDevice();

    // Download from Buffer
    void* mapPtr;
    auto downloadMapResult = device.mapMemory(memory, 0, data.size(), (vk::MemoryMapFlags)0, &mapPtr);
    if (downloadMapResult != vk::Result::eSuccess) {
        return downloadMapResult;
    }
    std::memcpy(data.data(), mapPtr, data.size());
    device.unmapMemory(memory);

    return vk::Result::eSuccess;
}

}  // namespace vkc::_hp
