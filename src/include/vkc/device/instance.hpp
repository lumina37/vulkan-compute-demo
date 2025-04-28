#pragma once

#include <expected>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/helper/error.hpp"

namespace vkc {

namespace rgs = std::ranges;

static constexpr std::string_view VALIDATION_LAYER_NAME{"VK_LAYER_KHRONOS_validation"};

class InstanceManager {
    InstanceManager(vk::Instance instance) noexcept;

public:
    InstanceManager(InstanceManager&& rhs) noexcept;
    ~InstanceManager() noexcept;

    [[nodiscard]] static std::expected<InstanceManager, Error> create() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getInstance(this Self&& self) noexcept {
        return std::forward_like<Self>(self).instance_;
    }

private:
    vk::Instance instance_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/instance.cpp"
#endif
