#pragma once

#include <cstdint>
#include <expected>

#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

std::expected<uint32_t, Error> defaultComputeQFamilyIndex(const PhysicalDeviceManager& phyDeviceMgr);

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/queue_family.cpp"
#endif
