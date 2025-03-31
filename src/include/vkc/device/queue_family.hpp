#pragma once

#include <cstdint>

#include "vkc/device/physical.hpp"

namespace vkc {

uint32_t defaultComputeQFamilyIndex(const PhysicalDeviceManager& phyDeviceMgr);

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/queue_family.cpp"
#endif
