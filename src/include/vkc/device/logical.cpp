#include <vulkan/vulkan.hpp>

#include "vkc/device/physical.hpp"
#include "vkc/device/queue_family.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/logical.hpp"
#endif

namespace vkc {

DeviceManager::DeviceManager(PhyDeviceManager& phyDeviceMgr, const QueueFamilyManager& queueFamilyMgr) {
    constexpr float priority = 1.0f;
    vk::DeviceQueueCreateInfo computeQueueInfo;
    computeQueueInfo.setQueuePriorities(priority);
    computeQueueInfo.setQueueFamilyIndex(queueFamilyMgr.getComputeQFamilyIndex());
    computeQueueInfo.setQueueCount(1);

    vk::DeviceCreateInfo deviceInfo;
    deviceInfo.setQueueCreateInfos(computeQueueInfo);

    auto& phyDevice = phyDeviceMgr.getPhysicalDevice();
    device_ = phyDevice.createDevice(deviceInfo);
}

DeviceManager::~DeviceManager() noexcept { device_.destroy(); }

}  // namespace vkc
