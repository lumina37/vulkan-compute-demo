#include <expected>

#include "vkc/device/props.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/set.hpp"
#endif

namespace vkc {

std::expected<float, Error> defaultJudge(const PhyDeviceWithProps_<PhyDeviceProps>& phyDeviceWithProps) noexcept {
    const PhyDeviceProps& props = phyDeviceWithProps.getProps();

    int deviceTypeAmp;
    switch (props.deviceType) {
        case vk::PhysicalDeviceType::eDiscreteGpu:
            deviceTypeAmp = 3;
            break;
        case vk::PhysicalDeviceType::eIntegratedGpu:
            deviceTypeAmp = 2;
            break;
        default:
            deviceTypeAmp = 1;
            break;
    }

    const float score = (float)(props.maxSharedMemSize * deviceTypeAmp);

    return score;
}

template class PhyDeviceSet_<PhyDeviceProps>;

}  // namespace vkc
