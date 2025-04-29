#include <cstdint>
#include <expected>
#include <utility>

#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/logical.hpp"
#endif

namespace vkc {

DeviceManager::DeviceManager(vk::Device device) noexcept : device_(device) {}

DeviceManager::DeviceManager(DeviceManager&& rhs) noexcept : device_(std::exchange(rhs.device_, nullptr)) {}

DeviceManager::~DeviceManager() noexcept {
    if (device_ == nullptr) return;
    device_.destroy();
    device_ = nullptr;
}

std::expected<DeviceManager, Error> DeviceManager::create(PhysicalDeviceManager& phyDeviceMgr,
                                                          uint32_t queueFamilyIdx) noexcept {
    constexpr float priority = 1.0f;
    vk::DeviceQueueCreateInfo computeQueueInfo;
    computeQueueInfo.setQueuePriorities(priority);
    computeQueueInfo.setQueueFamilyIndex(queueFamilyIdx);
    computeQueueInfo.setQueueCount(1);

    vk::DeviceCreateInfo deviceInfo;
    deviceInfo.setQueueCreateInfos(computeQueueInfo);

    auto& phyDevice = phyDeviceMgr.getPhysicalDevice();
    const auto [deviceRes, device] = phyDevice.createDevice(deviceInfo);
    if (deviceRes != vk::Result::eSuccess) {
        return std::unexpected{Error{deviceRes}};
    }

    return DeviceManager{device};
}

}  // namespace vkc
