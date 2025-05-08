#include <expected>

#include "vkc/device/props.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/set.hpp"
#endif

namespace vkc {

std::expected<float, Error> defaultJudge(const PhyDeviceWithProps_<PhyDeviceProps>& phyDeviceWithProps) noexcept {
    const PhyDeviceProps& props = phyDeviceWithProps.getPhyDeviceProps();

    float score = (float)props.maxSharedMemSize;

    if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) score *= 3.0f;
    if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) score *= 2.0f;

    return score;
}

template class PhyDeviceSet_<PhyDeviceProps>;

}  // namespace vkc
