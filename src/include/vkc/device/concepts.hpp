#pragma once

#include <concepts>

#include "vkc/device/physical/box.hpp"

namespace vkc {

template <typename Self>
concept CHasExtensionName = requires(Self self) {
    { self.extensionName } -> std::convertible_to<std::string_view>;
} || std::is_same_v<std::remove_cvref_t<Self>, vk::ExtensionProperties>;

template <typename Self>
concept CHasLayerName = requires(Self self) {
    { self.layerName } -> std::convertible_to<std::string_view>;
} || std::is_same_v<std::remove_cvref_t<Self>, vk::LayerProperties>;

template <typename Self>
concept CExt = CHasExtensionName<Self> || CHasLayerName<Self>;

}  // namespace vkc
