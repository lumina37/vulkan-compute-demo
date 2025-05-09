#include <expected>

#include "vkc/device/physical/props.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical/set.hpp"
#endif

namespace vkc {

std::expected<float, Error> defaultJudge(
    const PhyDeviceWithProps_<DefaultPhyDeviceProps>& phyDeviceWithProps) noexcept {
    const DefaultPhyDeviceProps& props = phyDeviceWithProps.getPhyDeviceProps();

    float score = (float)props.maxSharedMemSize;

    if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) score *= 3.0f;
    if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) score *= 2.0f;

    return score;
}

template class PhyDeviceSet_<DefaultPhyDeviceProps>;

}  // namespace vkc
