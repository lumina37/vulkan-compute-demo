#include <expected>

#include "vkc/device/props.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/set.hpp"
#endif

namespace vkc {

std::expected<float, Error> defaultJudge(const PhyDeviceWithProps_<PhyDeviceProps>& phyDeviceWithProps) noexcept {
    const PhyDeviceProps& props = phyDeviceWithProps.getProps();

    float score = props.subgroupSize;

    return score;
}

template class DeviceSet_<PhyDeviceProps>;

}  // namespace vkc
