#pragma once

#include <expected>

#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class PhyDeviceBox {
    PhyDeviceBox(vk::PhysicalDevice phyDevice) noexcept;

public:
    PhyDeviceBox(PhyDeviceBox&& rhs) noexcept = default;

    [[nodiscard]] static std::expected<PhyDeviceBox, Error> create(vk::PhysicalDevice phyDevice) noexcept;

    [[nodiscard]] vk::PhysicalDevice getPhyDevice() const noexcept { return phyDevice_; }

private:
    vk::PhysicalDevice phyDevice_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical/box.cpp"
#endif
