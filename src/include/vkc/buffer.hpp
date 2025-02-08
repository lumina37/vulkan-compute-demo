#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device.hpp"
#include "vkc/image.hpp"

namespace vkc {

class BufferManager {
public:
    inline BufferManager(const PhyDeviceManager& phyDeviceMgr, const DeviceManager& deviceMgr,
                         const vk::Extent2D& extent);
    inline ~BufferManager() noexcept;

    vk::DeviceSize size_;
    ImageManager srcImageMgr_;
    ImageManager dstImageMgr_;
    vk::Buffer srcStagingBuffer_;
    vk::Buffer dstStagingBuffer_;
    vk::DeviceMemory srcStagingMemory_;
    vk::DeviceMemory dstStagingMemory_;

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
};

inline BufferManager::BufferManager(const PhyDeviceManager& phyDeviceMgr, const DeviceManager& deviceMgr,
                                    const vk::Extent2D& extent)
    : size_(extent.width * extent.height),
      srcImageMgr_(phyDeviceMgr, deviceMgr, extent,
                   vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst),
      dstImageMgr_(phyDeviceMgr, deviceMgr, extent,
                   vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc),
      deviceMgr_(deviceMgr) {
    const auto& physicalDevice = phyDeviceMgr.getPhysicalDevice();
    const auto& device = deviceMgr.getDevice();
    createBuffer(physicalDevice, device, size_, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                 srcStagingBuffer_, srcStagingMemory_);
    createBuffer(physicalDevice, device, size_, vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                 dstStagingBuffer_, dstStagingMemory_);
}

BufferManager::~BufferManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyBuffer(srcStagingBuffer_);
    device.destroyBuffer(dstStagingBuffer_);
    device.freeMemory(srcStagingMemory_);
    device.freeMemory(dstStagingMemory_);
}

}  // namespace vkc
