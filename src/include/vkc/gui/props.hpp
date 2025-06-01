#pragma once

#include <expected>
#include <vector>

#include "vkc/device/physical/manager.hpp"
#include "vkc/gui/surface.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class DefaultSurfaceProps {
public:
    DefaultSurfaceProps() noexcept = default;
    DefaultSurfaceProps(const DefaultSurfaceProps&) = delete;
    DefaultSurfaceProps(DefaultSurfaceProps&&) noexcept = default;

    [[nodiscard]] static std::expected<DefaultSurfaceProps, Error> create(const PhyDeviceManager& phyDeviceMgr,
                                                                          const SurfaceManager& surfaceMgr) noexcept;

    // Members
    vk::Extent2D minExtent;
    vk::Extent2D maxExtent;
    vk::ImageUsageFlags supportImageUsage;
    vk::SurfaceTransformFlagsKHR supportTransform;
    vk::CompositeAlphaFlagsKHR supportCompositeAlpha;
    std::vector<vk::PresentModeKHR> presentModes;
    std::vector<vk::SurfaceFormatKHR> formats;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/props.cpp"
#endif
