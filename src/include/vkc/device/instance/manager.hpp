#pragma once

#include <expected>
#include <span>
#include <string>

#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

namespace rgs = std::ranges;

class InstanceManager {
    InstanceManager(vk::Instance instance) noexcept;

public:
    InstanceManager(InstanceManager&& rhs) noexcept;
    ~InstanceManager() noexcept;

    [[nodiscard]] static std::expected<InstanceManager, Error> create() noexcept;
    [[nodiscard]] static std::expected<InstanceManager, Error> createWithExts(
        std::span<const std::string_view> enableExtNames, std::span<const std::string_view> enableLayerNames) noexcept;

    [[nodiscard]] vk::Instance getInstance() const noexcept { return instance_; }

private:
    vk::Instance instance_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/instance/manager.cpp"
#endif
