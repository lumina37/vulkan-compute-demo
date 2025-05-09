#pragma once

#include <expected>
#include <utility>

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

    template <typename Self>
    [[nodiscard]] auto&& getInstance(this Self&& self) noexcept {
        return std::forward_like<Self>(self).instance_;
    }

private:
    vk::Instance instance_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/instance/manager.cpp"
#endif
