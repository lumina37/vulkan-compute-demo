#pragma once

#include <expected>
#include <utility>

#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class PhyDeviceManager {
    PhyDeviceManager(vk::PhysicalDevice phyDevice) noexcept;

public:
    PhyDeviceManager(PhyDeviceManager&& rhs) noexcept = default;

    [[nodiscard]] static std::expected<PhyDeviceManager, Error> create(vk::PhysicalDevice phyDevice) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPhyDevice(this Self&& self) noexcept {
        return std::forward_like<Self>(self).phyDevice_;
    }

private:
    vk::PhysicalDevice phyDevice_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical.cpp"
#endif
