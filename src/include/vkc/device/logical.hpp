#pragma once

#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/physical.hpp"
#include "vkc/queue_family.hpp"

namespace vkc {

class DeviceManager {
public:
    DeviceManager(PhyDeviceManager& phyDeviceMgr, const QueueFamilyManager& queueFamilyMgr);
    ~DeviceManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDevice(this Self&& self) noexcept {
        return std::forward_like<Self>(self).device_;
    }

private:
    vk::Device device_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/logical.cpp"
#endif
