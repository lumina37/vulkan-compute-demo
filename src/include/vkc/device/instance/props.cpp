#include <expected>
#include <ranges>

#include "vkc/device/extensions.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/instance/props.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

std::expected<DefaultInstanceProps, Error> DefaultInstanceProps::create() noexcept {
    DefaultInstanceProps props;

    auto [extPropsRes, extProps] = vk::enumerateInstanceExtensionProperties();
    if (extPropsRes != vk::Result::eSuccess) {
        return std::unexpected{Error{extPropsRes}};
    }

    auto extEntriesRes = ExtEntries_<vk::ExtensionProperties>::create(std::move(extProps));
    if (!extEntriesRes) return std::unexpected{std::move(extEntriesRes.error())};
    props.exts = std::move(extEntriesRes.value());

    auto [layerPropsRes, layerProps] = vk::enumerateInstanceLayerProperties();
    if (layerPropsRes != vk::Result::eSuccess) {
        return std::unexpected{Error{layerPropsRes}};
    }

    auto layerEntriesRes = ExtEntries_<vk::LayerProperties>::create(std::move(layerProps));
    if (!layerEntriesRes) return std::unexpected{std::move(layerEntriesRes.error())};
    props.layers = std::move(layerEntriesRes.value());

    return props;
}

}  // namespace vkc
