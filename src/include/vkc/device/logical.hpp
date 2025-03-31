#pragma once

#include <cstdint>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/physical.hpp"

namespace vkc {

class DeviceManager {
public:
    DeviceManager(PhysicalDeviceManager& phyDeviceMgr, uint32_t queueFamilyIdx);
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
