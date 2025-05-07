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

    const auto& phyDeviceProps = propsChain.get<vk::PhysicalDeviceProperties2>().properties;
    props.apiVersion = phyDeviceProps.apiVersion;
    props.deviceType = phyDeviceProps.deviceType;
    props.maxSharedMemSize = phyDeviceProps.limits.maxComputeSharedMemorySize;
    props.timestampPeriod = phyDeviceProps.limits.timestampPeriod;
    props.supportTimeQueryForAllQueue = (bool)phyDeviceProps.limits.timestampComputeAndGraphics;

    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceShaderFloat16Int8Features> featureChain;
    phyDevice.getFeatures2(&featureChain.get());

    const auto& shaderFp16Int8Features = featureChain.get<vk::PhysicalDeviceShaderFloat16Int8Features>();
    props.supportFp16 = (bool)shaderFp16Int8Features.shaderFloat16;

    return props;
}

template class PhyDeviceWithProps_<PhyDeviceProps>;

}  // namespace vkc
