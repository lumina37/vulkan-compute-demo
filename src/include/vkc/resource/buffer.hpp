#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/resource/image.hpp"

namespace vkc {

class BufferManager {
public:
    inline BufferManager(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, const ExtentManager& extent);

    template <typename Self>
    [[nodiscard]] auto&& getSrcImageMgr(this Self&& self) noexcept {
        return std::forward_like<Self>(self).srcImageMgr_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getDstImageMgr(this Self&& self) noexcept {
        return std::forward_like<Self>(self).dstImageMgr_;
    }

private:
    ImageManager srcImageMgr_;
    ImageManager dstImageMgr_;
};

inline BufferManager::BufferManager(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr,
                                    const ExtentManager& extent)
    : srcImageMgr_(phyDeviceMgr, deviceMgr, extent,
                   vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                   vk::DescriptorType::eSampledImage),
      dstImageMgr_(phyDeviceMgr, deviceMgr, extent,
                   vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
                   vk::DescriptorType::eStorageImage) {}

}  // namespace vkc
