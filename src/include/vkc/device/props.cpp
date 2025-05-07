#include <expected>
#include <ranges>

#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/props.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

std::expected<PhyDeviceProps, Error> PhyDeviceProps::create(const PhyDeviceManager& phyDeviceMgr) noexcept {
    PhyDeviceProps props;
    const auto phyDevice = phyDeviceMgr.getPhyDevice();

    vk::StructureChain<vk::PhysicalDeviceProperties2> propsChain;
    phyDevice.getProperties2(&propsChain.get());

    const auto& deviceProps = propsChain.get<vk::PhysicalDeviceProperties2>().properties;
    props.deviceType = deviceProps.deviceType;
    props.maxSharedMemSize = deviceProps.limits.maxComputeSharedMemorySize;
    props.timestampPeriod = deviceProps.limits.timestampPeriod;
    props.supportTimeQueryForAllQueue = (bool)deviceProps.limits.timestampComputeAndGraphics;

    return props;
}

template class PhyDeviceWithProps_<PhyDeviceProps>;

}  // namespace vkc
