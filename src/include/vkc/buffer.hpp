#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device.hpp"
#include "vkc/image.hpp"

namespace vkc {

class BufferManager {
public:
    inline BufferManager(const PhyDeviceManager& phyDeviceMgr, const DeviceManager& deviceMgr,
                         const vk::Extent2D& extent);

    vk::DeviceSize size_;
    ImageManager srcImageMgr_;
    ImageManager dstImageMgr_;
};

inline BufferManager::BufferManager(const PhyDeviceManager& phyDeviceMgr, const DeviceManager& deviceMgr,
                                    const vk::Extent2D& extent)
    : size_(extent.width * extent.height),
      srcImageMgr_(phyDeviceMgr, deviceMgr, extent,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                   vk::DescriptorType::eSampledImage),
      dstImageMgr_(phyDeviceMgr, deviceMgr, extent,
                   vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                   vk::DescriptorType::eStorageImage) {}

}  // namespace vkc
