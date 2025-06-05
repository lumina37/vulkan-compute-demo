#include <expected>

#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical/box.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

PhyDeviceBox::PhyDeviceBox(vk::PhysicalDevice phyDevice) noexcept : phyDevice_(phyDevice) {}

std::expected<PhyDeviceBox, Error> PhyDeviceBox::create(vk::PhysicalDevice phyDevice) noexcept {
    return PhyDeviceBox{phyDevice};
}

}  // namespace vkc
