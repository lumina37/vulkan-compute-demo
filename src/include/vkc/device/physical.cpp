#include <print>

#include <vulkan/vulkan.hpp>

#include "vkc/helper/defines.hpp"
#include "vkc/instance.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical.hpp"
#endif

namespace vkc {

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
                std::println("Selected physical device: {}", phyDeviceProp.deviceName.data());
                std::println("Vulkan API version: {}.{}.{}", VK_API_VERSION_MAJOR(phyDeviceProp.apiVersion),
                             VK_API_VERSION_MINOR(phyDeviceProp.apiVersion),
                             VK_API_VERSION_PATCH(phyDeviceProp.apiVersion));
            }
            break;
        }
    }
}

}  // namespace vkc
