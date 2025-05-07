#pragma once

#include <cstdint>
#include <expected>
#include <utility>

#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class DeviceManager {
    DeviceManager(vk::Device device) noexcept;

public:
    DeviceManager(DeviceManager&& rhs) noexcept;
    ~DeviceManager() noexcept;

    [[nodiscard]] static std::expected<DeviceManager, Error> create(PhyDeviceManager& phyDeviceMgr,
                                                                    uint32_t queueFamilyIdx) noexcept;

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
