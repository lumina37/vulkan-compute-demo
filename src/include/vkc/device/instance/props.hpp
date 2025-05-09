#pragma once

#include <expected>

#include "vkc/device/extensions.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

class DefaultInstanceProps {
public:
    DefaultInstanceProps() noexcept = default;
    DefaultInstanceProps(const DefaultInstanceProps&) = delete;
    DefaultInstanceProps(DefaultInstanceProps&&) noexcept = default;

    [[nodiscard]] static std::expected<DefaultInstanceProps, Error> create() noexcept;

    // Members
    ExtEntries_<vk::ExtensionProperties> exts;
    ExtEntries_<vk::LayerProperties> layers;
};

}  // namespace vkc

#ifdef _vkc_LIB_HEADER_ONLY
#    include "vkc/device/instance/props.cpp"
#endif
