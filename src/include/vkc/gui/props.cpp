#include <expected>
#include <ranges>
#include <utility>

#include "vkc/device/physical/manager.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/props.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

std::expected<DefaultSurfaceProps, Error> DefaultSurfaceProps::create(const PhyDeviceManager& phyDeviceMgr,
                                                                      const SurfaceManager& surfaceMgr) noexcept {
    DefaultSurfaceProps props;
    const auto phyDevice = phyDeviceMgr.getPhyDevice();
    const auto surface = surfaceMgr.getSurface();

    auto surfaceCapsRes = phyDevice.getSurfaceCapabilitiesKHR(surface);
    if (surfaceCapsRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{surfaceCapsRes.result}};
    }
    const auto& surfaceCaps = surfaceCapsRes.value;
    props.minExtent = surfaceCaps.minImageExtent;
    props.maxExtent = surfaceCaps.maxImageExtent;
    props.supportImageUsage = surfaceCaps.supportedUsageFlags;
    props.supportTransform = surfaceCaps.supportedTransforms;
    props.supportCompositeAlpha = surfaceCaps.supportedCompositeAlpha;

    auto surfacePresentModesRes = phyDevice.getSurfacePresentModesKHR(surface);
    if (surfacePresentModesRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{surfaceCapsRes.result}};
    }
    auto& surfacePresentModes = surfacePresentModesRes.value;
    props.presentModes = std::move(surfacePresentModes);

    auto surfaceFormatsRes = phyDevice.getSurfaceFormatsKHR(surface);
    if (surfaceFormatsRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{surfaceCapsRes.result}};
    }
    auto& surfaceFormats = surfaceFormatsRes.value;
    props.formats = std::move(surfaceFormats);

    return props;
}

}  // namespace vkc
