#pragma once

#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/instance.hpp"

namespace vkc {

class PhyDeviceManager {
public:
    PhyDeviceManager(const InstanceManager& instMgr);

    template <typename Self>
    [[nodiscard]] auto&& getPhysicalDevice(this Self&& self) noexcept {
        return std::forward_like<Self>(self).physicalDevice_;
    }

    [[nodiscard]] float getTimestampPeriod() const noexcept { return limits_.timestampPeriod; }

private:
    vk::PhysicalDevice physicalDevice_;
    vk::PhysicalDeviceLimits limits_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical.cpp"
#endif
