#include <utility>

#include "vkc/device/physical/box.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/gui/props.hpp"
#endif

namespace vkc {

std::expected<DefaultSurfaceProps, Error> DefaultSurfaceProps::create(const PhyDeviceBox& phyDeviceBox,
                                                                      const SurfaceBox& surfaceBox) noexcept {
    DefaultSurfaceProps props;
    const vk::PhysicalDevice phyDevice = phyDeviceBox.getPhyDevice();
    const vk::SurfaceKHR surface = surfaceBox.getSurface();

    auto surfaceCapsRes = phyDevice.getSurfaceCapabilitiesKHR(surface);
    if (surfaceCapsRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, surfaceCapsRes.result}};
    }
    const auto& surfaceCaps = surfaceCapsRes.value;
    props.minExtent = surfaceCaps.minImageExtent;
    props.maxExtent = surfaceCaps.maxImageExtent;
    props.supportImageUsage = surfaceCaps.supportedUsageFlags;
    props.supportTransform = surfaceCaps.supportedTransforms;
    props.supportCompositeAlpha = surfaceCaps.supportedCompositeAlpha;

    auto surfacePresentModesRes = phyDevice.getSurfacePresentModesKHR(surface);
    if (surfacePresentModesRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, surfaceCapsRes.result}};
    }
    auto& surfacePresentModes = surfacePresentModesRes.value;
    props.presentModes = std::move(surfacePresentModes);

    auto surfaceFormatsRes = phyDevice.getSurfaceFormatsKHR(surface);
    if (surfaceFormatsRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, surfaceCapsRes.result}};
    }
    auto& surfaceFormats = surfaceFormatsRes.value;
    props.formats = std::move(surfaceFormats);

    return props;
}

}  // namespace vkc
