#include <cstdint>
#include <expected>
#include <ranges>
#include <tuple>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/memory.hpp"
#endif

namespace vkc::_hp {

namespace rgs = std::ranges;

std::expected<uint32_t, Error> findMemoryTypeIdx(const PhyDeviceManager& phyDeviceMgr, const uint32_t supportedMemType,
                                                 const vk::MemoryPropertyFlags memProps) noexcept {
    const auto& physicalDevice = phyDeviceMgr.getPhyDevice();
    const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (const auto [idx, memType] : rgs::views::enumerate(memProperties.memoryTypes)) {
        const bool isSupported = supportedMemType & (1 << idx);
        const bool isSufficient = (memType.propertyFlags & memProps) == memProps;
        if (isSupported && isSufficient) {
            return (uint32_t)idx;
        }
    }

    return std::unexpected{Error{-1, "no sufficient memory type"}};
}

std::expected<void, Error> allocBufferMemory(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr,
                                             vk::Buffer& buffer, vk::MemoryPropertyFlags memProps,
                                             vk::DeviceMemory& bufferMemory) noexcept {
    vk::Device device = deviceMgr.getDevice();

    vk::MemoryAllocateInfo allocInfo;
    const vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
    allocInfo.setAllocationSize(memRequirements.size);
    auto memoryTypeIndexRes = findMemoryTypeIdx(phyDeviceMgr, memRequirements.memoryTypeBits, memProps);
    if (!memoryTypeIndexRes) return std::unexpected{std::move(memoryTypeIndexRes.error())};
    allocInfo.memoryTypeIndex = memoryTypeIndexRes.value();

    vk::Result bufferMemoryRes;
    std::tie(bufferMemoryRes, bufferMemory) = device.allocateMemory(allocInfo);
    if (bufferMemoryRes != vk::Result::eSuccess) {
        return std::unexpected{Error{bufferMemoryRes}};
    }

    return {};
}

std::expected<void, Error> allocImageMemory(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr,
                                            vk::Image& image, vk::MemoryPropertyFlags memProps,
                                            vk::DeviceMemory& bufferMemory) noexcept {
    vk::Device device = deviceMgr.getDevice();

    vk::MemoryAllocateInfo allocInfo;
    const vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);
    allocInfo.setAllocationSize(memRequirements.size);
    auto memoryTypeIndexRes = findMemoryTypeIdx(phyDeviceMgr, memRequirements.memoryTypeBits, memProps);
    if (!memoryTypeIndexRes) return std::unexpected{std::move(memoryTypeIndexRes.error())};
    allocInfo.memoryTypeIndex = memoryTypeIndexRes.value();

    vk::Result bufferMemoryRes;
    std::tie(bufferMemoryRes, bufferMemory) = device.allocateMemory(allocInfo);
    if (bufferMemoryRes != vk::Result::eSuccess) {
        return std::unexpected{Error{bufferMemoryRes}};
    }

    return {};
}

std::expected<void, Error> uploadFrom(DeviceManager& deviceMgr, vk::DeviceMemory& memory,
                                      const std::span<const std::byte> data) noexcept {
    vk::Device device = deviceMgr.getDevice();

    // Upload to Buffer
    void* mapPtr;
    auto uploadMapRes = device.mapMemory(memory, 0, data.size(), (vk::MemoryMapFlags)0, &mapPtr);
    if (uploadMapRes != vk::Result::eSuccess) {
        return std::unexpected{Error{uploadMapRes}};
    }

    std::memcpy(mapPtr, data.data(), data.size());
    device.unmapMemory(memory);

    return {};
}

std::expected<void, Error> downloadTo(DeviceManager& deviceMgr, const vk::DeviceMemory& memory,
                                      const std::span<std::byte> data) noexcept {
    vk::Device device = deviceMgr.getDevice();

    // Download from Buffer
    void* mapPtr;
    auto downloadMapRes = device.mapMemory(memory, 0, data.size(), (vk::MemoryMapFlags)0, &mapPtr);
    if (downloadMapRes != vk::Result::eSuccess) {
        return std::unexpected{Error{downloadMapRes}};
    }
    std::memcpy(data.data(), mapPtr, data.size());
    device.unmapMemory(memory);

    return {};
}

}  // namespace vkc::_hp
