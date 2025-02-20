#pragma once

#include <utility>

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
    constexpr float priority = 1.0f;
    vk::DeviceQueueCreateInfo computeQueueInfo;
    computeQueueInfo.setQueuePriorities(priority);
    computeQueueInfo.setQueueFamilyIndex(queueFamilyMgr.getComputeQFamilyIndex());
    computeQueueInfo.setQueueCount(1);

    vk::DeviceCreateInfo deviceInfo;
    deviceInfo.setQueueCreateInfos(computeQueueInfo);

    const auto& phyDevice = phyDeviceMgr.getPhysicalDevice();
    device_ = phyDevice.createDevice(deviceInfo);
}

DeviceManager::~DeviceManager() noexcept { device_.destroy(); }

}  // namespace vkc
