#pragma once

#include <expected>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/instance.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

class PhysicalDeviceManager {
    PhysicalDeviceManager(vk::PhysicalDevice&& physicalDevice, vk::PhysicalDeviceLimits&& limits) noexcept;

public:
    PhysicalDeviceManager(PhysicalDeviceManager&& rhs) noexcept;

    [[nodiscard]] static std::expected<PhysicalDeviceManager, Error> create(const InstanceManager& instMgr) noexcept;

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
