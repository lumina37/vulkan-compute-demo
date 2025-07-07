#include <cstdint>
#include <expected>
#include <ranges>
#include <tuple>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/memory.hpp"
#endif

namespace vkc::_hp {

namespace rgs = std::ranges;

std::expected<uint32_t, Error> findMemoryTypeIdx(const PhyDeviceBox& phyDeviceBox, const uint32_t supportedMemType,
                                                 const vk::MemoryPropertyFlags memProps) noexcept {
    const vk::PhysicalDevice physicalDevice = phyDeviceBox.getPhyDevice();
    const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (const auto [idx, memType] : rgs::views::enumerate(memProperties.memoryTypes)) {
        const bool isSupported = supportedMemType & (1 << idx);
        const bool isSufficient = (memType.propertyFlags & memProps) == memProps;
        if (isSupported && isSufficient) {
            return (uint32_t)idx;
        }
    }

    return std::unexpected{Error{ECate::eVkC, ECode::eResourceInvalid, "no sufficient memory type"}};
}

std::expected<void, Error> allocBufferMemory(const PhyDeviceBox& phyDeviceBox, DeviceBox& deviceBox, vk::Buffer& buffer,
                                             const vk::MemoryPropertyFlags memProps,
                                             vk::DeviceMemory& bufferMemory) noexcept {
    vk::Device device = deviceBox.getDevice();

    vk::MemoryAllocateInfo allocInfo;
    const vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
    allocInfo.setAllocationSize(memRequirements.size);
    auto memoryTypeIndexRes = findMemoryTypeIdx(phyDeviceBox, memRequirements.memoryTypeBits, memProps);
    if (!memoryTypeIndexRes) return std::unexpected{std::move(memoryTypeIndexRes.error())};
    allocInfo.memoryTypeIndex = memoryTypeIndexRes.value();

    vk::Result bufferMemoryRes;
    std::tie(bufferMemoryRes, bufferMemory) = device.allocateMemory(allocInfo);
    if (bufferMemoryRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bufferMemoryRes}};
    }

    return {};
}

std::expected<void, Error> allocImageMemory(const PhyDeviceBox& phyDeviceBox, DeviceBox& deviceBox, vk::Image& image,
                                            const vk::MemoryPropertyFlags memProps,
                                            vk::DeviceMemory& bufferMemory) noexcept {
    vk::Device device = deviceBox.getDevice();

    vk::MemoryAllocateInfo allocInfo;
    const vk::MemoryRequirements memRequirements = device.getImageMemoryRequirements(image);
    allocInfo.setAllocationSize(memRequirements.size);
    auto memoryTypeIndexRes = findMemoryTypeIdx(phyDeviceBox, memRequirements.memoryTypeBits, memProps);
    if (!memoryTypeIndexRes) return std::unexpected{std::move(memoryTypeIndexRes.error())};
    allocInfo.memoryTypeIndex = memoryTypeIndexRes.value();

    vk::Result bufferMemoryRes;
    std::tie(bufferMemoryRes, bufferMemory) = device.allocateMemory(allocInfo);
    if (bufferMemoryRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, bufferMemoryRes}};
    }

    return {};
}

MemMapBox::MemMapBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DeviceMemory memory, void* mapPtr) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), memory_(memory), mapPtr_(mapPtr) {}

MemMapBox::MemMapBox(MemMapBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      memory_(std::exchange(rhs.memory_, nullptr)),
      mapPtr_(std::exchange(rhs.mapPtr_, nullptr)) {}

MemMapBox::~MemMapBox() noexcept {
    if (mapPtr_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.unmapMemory(memory_);
    mapPtr_ = nullptr;
    memory_ = nullptr;
}

std::expected<MemMapBox, Error> MemMapBox::create(std::shared_ptr<DeviceBox> pDeviceBox, vk::DeviceMemory& memory,
                                                  const size_t size) noexcept {
    vk::Device device = pDeviceBox->getDevice();

    void* mapPtr;
    const auto mapRes = device.mapMemory(memory, 0, size, (vk::MemoryMapFlags)0, &mapPtr);
    if (mapRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, mapRes}};
    }

    return MemMapBox{std::move(pDeviceBox), memory, mapPtr};
}

}  // namespace vkc::_hp
