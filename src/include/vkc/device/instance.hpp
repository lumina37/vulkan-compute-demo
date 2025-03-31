#pragma once

#include <utility>

#include <vulkan/vulkan.hpp>

namespace vkc {

namespace rgs = std::ranges;

static const char* VALIDATION_LAYER_NAME = "VK_LAYER_KHRONOS_validation";

class InstanceManager {
public:
    InstanceManager();
    ~InstanceManager() noexcept;

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
