#include <cstdint>
#include <expected>
#include <ranges>
#include <tuple>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/memory.hpp"
#endif

namespace vkc {

MemoryBox::MemoryBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::DeviceMemory memory,
                     const vk::MemoryRequirements& requirements) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), memory_(memory), requirements_(requirements) {}

MemoryBox::MemoryBox(MemoryBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      memory_(std::exchange(rhs.memory_, nullptr)),
      requirements_(rhs.requirements_) {}

MemoryBox::~MemoryBox() noexcept {
    if (memory_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.freeMemory(memory_);
}

std::expected<MemoryBox, Error> MemoryBox::createByIndex(std::shared_ptr<DeviceBox> pDeviceBox,
                                                         const vk::MemoryRequirements& requirements,
                                                         uint32_t memTypeIndex) noexcept {
    vk::Device device = pDeviceBox->getDevice();

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.setAllocationSize(requirements.size);
    allocInfo.setMemoryTypeIndex(memTypeIndex);

    vk::Result memoryRes;
    vk::DeviceMemory memory;
    std::tie(memoryRes, memory) = device.allocateMemory(allocInfo);
    if (memoryRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, memoryRes}};
    }

    return MemoryBox{std::move(pDeviceBox), memory, requirements};
}

std::expected<MemoryBox, Error> MemoryBox::create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                  const vk::MemoryRequirements& requirements,
                                                  vk::MemoryPropertyFlags props) noexcept {
    vk::PhysicalDevice phyDevice = pDeviceBox->getPhyDevice();

    auto memTypeIndexRes = _hp::findMemoryTypeIdx(phyDevice, requirements.memoryTypeBits, props);
    if (!memTypeIndexRes) return std::unexpected{std::move(memTypeIndexRes.error())};
    const uint32_t memTypeIndex = memTypeIndexRes.value();

    return createByIndex(std::move(pDeviceBox), requirements, memTypeIndex);
}

std::expected<void*, Error> MemoryBox::memMap() noexcept {
    vk::Device device = pDeviceBox_->getDevice();

    void* mapPtr;
    const auto mapRes = device.mapMemory(memory_, 0, requirements_.size, (vk::MemoryMapFlags)0, &mapPtr);
    if (mapRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, mapRes}};
    }

    return mapPtr;
}

void MemoryBox::memUnmap() noexcept {
    vk::Device device = pDeviceBox_->getDevice();
    device.unmapMemory(memory_);
}

namespace _hp {

namespace rgs = std::ranges;

vk::MemoryRequirements getMemoryRequirements(const DeviceBox& deviceBox, vk::Image image) noexcept {
    vk::Device device = deviceBox.getDevice();
    vk::MemoryRequirements requirements = device.getImageMemoryRequirements(image);
    return requirements;
}

vk::MemoryRequirements getMemoryRequirements(const DeviceBox& deviceBox, vk::Buffer buffer) noexcept {
    vk::Device device = deviceBox.getDevice();
    vk::MemoryRequirements requirements = device.getBufferMemoryRequirements(buffer);
    return requirements;
}

std::expected<uint32_t, Error> findMemoryTypeIdx(vk::PhysicalDevice physicalDevice, const uint32_t supportedMemType,
                                                 const vk::MemoryPropertyFlags memProps) noexcept {
    const vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (const auto [idx, memType] : rgs::views::enumerate(memProperties.memoryTypes)) {
        const bool isSupported = supportedMemType & (1 << idx);
        const bool isSufficient = (memType.propertyFlags & memProps) == memProps;
        if (isSupported && isSufficient) {
            return (uint32_t)idx;
        }
    }

    return std::unexpected{Error{ECate::eVkC, ECode::eNoSupport, "no sufficient memory type"}};
}

}  // namespace _hp

}  // namespace vkc
