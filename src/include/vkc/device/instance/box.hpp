#pragma once

#include <span>

#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class InstanceBox {
    InstanceBox(vk::Instance instance) noexcept;

public:
    InstanceBox(const InstanceBox&) = delete;
    InstanceBox(InstanceBox&& rhs) noexcept;
    ~InstanceBox() noexcept;

    [[nodiscard]] static std::expected<InstanceBox, Error> create() noexcept;
    [[nodiscard]] static std::expected<InstanceBox, Error> createWithExts(
        std::span<const std::string_view> enableExtNames, std::span<const std::string_view> enableLayerNames) noexcept;

    [[nodiscard]] vk::Instance getInstance() const noexcept { return instance_; }

private:
    vk::Instance instance_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/instance/box.cpp"
#endif
