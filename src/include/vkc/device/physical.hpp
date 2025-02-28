#pragma once

#include <print>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/helper/defines.hpp"
#include "vkc/instance.hpp"

namespace vkc {

class PhyDeviceManager {
public:
    inline PhyDeviceManager(const InstanceManager& instMgr);

    template <typename Self>
    [[nodiscard]] auto&& getPhysicalDevice(this Self&& self) noexcept {
        return std::forward_like<Self>(self).physicalDevice_;
    }

    [[nodiscard]] inline float getTimestampPeriod() const noexcept { return limits_.timestampPeriod; }

private:
    vk::PhysicalDevice physicalDevice_;
    vk::PhysicalDeviceLimits limits_;
};

PhyDeviceManager::PhyDeviceManager(const InstanceManager& instMgr) {
    const auto& instance = instMgr.getInstance();

    const auto isPhysicalDeviceSuitable = [](const vk::PhysicalDeviceLimits& phyDeviceLimits) {
        if constexpr (ENABLE_DEBUG) {
            if (phyDeviceLimits.timestampPeriod == 0) return false;
            if (!phyDeviceLimits.timestampComputeAndGraphics) return false;
        }

        return true;
    };

    const auto& physicalDevices = instance.enumeratePhysicalDevices();

    for (const auto& physicalDevice : physicalDevices) {
        const auto& phyDeviceProp = physicalDevice.getProperties();
        const auto& limits = phyDeviceProp.limits;
        if (isPhysicalDeviceSuitable(limits)) {
            physicalDevice_ = physicalDevice;
            limits_ = limits;
            if constexpr (ENABLE_DEBUG) {
                std::println("Init physical device: {}", phyDeviceProp.deviceName.data());
            }
            break;
        }
    }
}

}  // namespace vkc
