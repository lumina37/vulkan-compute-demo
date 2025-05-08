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

std::expected<DefaultPhyDeviceProps, Error> DefaultPhyDeviceProps::create(
    const PhyDeviceManager& phyDeviceMgr) noexcept {
    DefaultPhyDeviceProps retProps;
    const auto phyDevice = phyDeviceMgr.getPhyDevice();

    vk::StructureChain<vk::PhysicalDeviceProperties2> propsChain;
    phyDevice.getProperties2(&propsChain.get());

    const auto& phyDeviceProps2 = propsChain.get<vk::PhysicalDeviceProperties2>().properties;
    retProps.apiVersion = phyDeviceProps2.apiVersion;
    retProps.deviceType = phyDeviceProps2.deviceType;
    retProps.maxSharedMemSize = phyDeviceProps2.limits.maxComputeSharedMemorySize;
    retProps.timestampPeriod = phyDeviceProps2.limits.timestampPeriod;
    retProps.supportTimeQueryForAllQueue = (bool)phyDeviceProps2.limits.timestampComputeAndGraphics;

    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceShaderFloat16Int8Features> featureChain;
    phyDevice.getFeatures2(&featureChain.get());

    const auto& shaderFp16Int8Features = featureChain.get<vk::PhysicalDeviceShaderFloat16Int8Features>();
    retProps.supportFp16 = (bool)shaderFp16Int8Features.shaderFloat16;

    return retProps;
}

template class PhyDeviceWithProps_<DefaultPhyDeviceProps>;

}  // namespace vkc
