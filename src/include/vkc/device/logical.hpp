#pragma once

#include <utility>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vkc/device/physical.hpp"
#include "vkc/queue_family.hpp"

namespace vkc {

class DeviceManager {
public:
    inline DeviceManager(const PhyDeviceManager& phyDeviceMgr, const QueueFamilyManager& queueFamilyMgr);
    inline ~DeviceManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDevice(this Self& self) noexcept {
        return std::forward_like<Self>(self).device_;
    }

private:
    vk::Device device_;
};

DeviceManager::DeviceManager(const PhyDeviceManager& phyDeviceMgr, const QueueFamilyManager& queueFamilyMgr) {
    const auto& phyDevice = phyDeviceMgr.getPhysicalDevice();

    constexpr float priority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> deviceQueueInfos;
    vk::DeviceQueueCreateInfo graphicsQueueInfo;
    graphicsQueueInfo.setQueuePriorities(priority);
    graphicsQueueInfo.setQueueFamilyIndex(queueFamilyMgr.getComputeQFamilyIndex());
    graphicsQueueInfo.setQueueCount(1);
    deviceQueueInfos.push_back(graphicsQueueInfo);

    vk::DeviceCreateInfo deviceInfo;
    deviceInfo.setQueueCreateInfos(deviceQueueInfos);

    device_ = phyDevice.createDevice(deviceInfo);
}

DeviceManager::~DeviceManager() noexcept { device_.destroy(); }

}  // namespace vkc
