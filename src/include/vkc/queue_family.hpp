#pragma once

#include <cstdint>

#include "vkc/device/physical.hpp"

namespace vkc {

class QueueFamilyManager {
public:
    QueueFamilyManager(const PhyDeviceManager& phyDeviceMgr);

    [[nodiscard]] uint32_t getComputeQFamilyIndex() const noexcept { return computeQFamilyIndex_; }

private:
    uint32_t computeQFamilyIndex_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/queue_family.cpp"
#endif
