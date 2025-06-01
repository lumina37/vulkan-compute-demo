#include <expected>
#include <ranges>

#include "vkc/device/physical/manager.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical/props.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

std::expected<DefaultPhyDeviceProps, Error> DefaultPhyDeviceProps::create(
    const PhyDeviceManager& phyDeviceMgr) noexcept {
    DefaultPhyDeviceProps props;
    const vk::PhysicalDevice phyDevice = phyDeviceMgr.getPhyDevice();

    auto [extPropsRes, extProps] = phyDevice.enumerateDeviceExtensionProperties();
    if (extPropsRes != vk::Result::eSuccess) {
        return std::unexpected{Error{extPropsRes}};
    }

    auto extEntriesRes = ExtEntries_<vk::ExtensionProperties>::create(std::move(extProps));
    if (!extEntriesRes) return std::unexpected{std::move(extEntriesRes.error())};
    props.extensions = std::move(extEntriesRes.value());

    vk::StructureChain<vk::PhysicalDeviceProperties2> propsChain;
    phyDevice.getProperties2(&propsChain.get());

    const auto& phyDeviceProps2 = propsChain.get<vk::PhysicalDeviceProperties2>().properties;
    props.apiVersion = phyDeviceProps2.apiVersion;
    props.deviceType = phyDeviceProps2.deviceType;
    props.maxSharedMemSize = phyDeviceProps2.limits.maxComputeSharedMemorySize;
    props.timestampPeriod = phyDeviceProps2.limits.timestampPeriod;
    props.supportTimeQuery = (bool)phyDeviceProps2.limits.timestampComputeAndGraphics;

    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceShaderFloat16Int8Features> featureChain;
    phyDevice.getFeatures2(&featureChain.get());

    const auto& shaderFp16Int8Features = featureChain.get<vk::PhysicalDeviceShaderFloat16Int8Features>();
    props.supportFp16 = (bool)shaderFp16Int8Features.shaderFloat16;

    return props;
}

std::expected<float, Error> DefaultPhyDeviceProps::score() const noexcept {
    float score = (float)maxSharedMemSize;

    if (deviceType == vk::PhysicalDeviceType::eDiscreteGpu) score *= 3.0f;
    if (deviceType == vk::PhysicalDeviceType::eIntegratedGpu) score *= 2.0f;

    return score;
}

template class PhyDeviceWithProps_<DefaultPhyDeviceProps>;

}  // namespace vkc
