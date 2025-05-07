#include <expected>

#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

PhyDeviceManager::PhyDeviceManager(vk::PhysicalDevice phyDevice) noexcept : phyDevice_(phyDevice) {}

std::expected<PhyDeviceManager, Error> PhyDeviceManager::create(vk::PhysicalDevice phyDevice) noexcept {
    return PhyDeviceManager{phyDevice};
}

}  // namespace vkc
