#include <cstdint>

#include <vulkan/vulkan.hpp>

#include "vkc/device/physical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/logical.hpp"
#endif

namespace vkc {

DeviceManager::DeviceManager(PhysicalDeviceManager& phyDeviceMgr, const uint32_t queueFamilyIdx) {
    constexpr float priority = 1.0f;
    vk::DeviceQueueCreateInfo computeQueueInfo;
    computeQueueInfo.setQueuePriorities(priority);
    computeQueueInfo.setQueueFamilyIndex(queueFamilyIdx);
    computeQueueInfo.setQueueCount(1);

    vk::DeviceCreateInfo deviceInfo;
    deviceInfo.setQueueCreateInfos(computeQueueInfo);

    auto& phyDevice = phyDeviceMgr.getPhysicalDevice();
    device_ = phyDevice.createDevice(deviceInfo);
}

DeviceManager::~DeviceManager() noexcept { device_.destroy(); }

}  // namespace vkc
