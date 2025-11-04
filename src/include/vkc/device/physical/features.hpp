#pragma once

#include "vkc/device/physical/box.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <typename... TFeat>
class PhyDeviceFeatures_ {
public:
    PhyDeviceFeatures_() noexcept = default;
    PhyDeviceFeatures_(const PhyDeviceFeatures_&) = delete;
    PhyDeviceFeatures_(PhyDeviceFeatures_&&) noexcept = default;

    [[nodiscard]] static std::expected<PhyDeviceFeatures_, Error> create(const PhyDeviceBox& phyDeviceBox) noexcept;

    [[nodiscard]] vk::PhysicalDeviceFeatures2* getPFeature() noexcept { return &featureChain_.get(); }

private:
    vk::StructureChain<TFeat...> featureChain_;
};

template <typename... TFeat>
auto PhyDeviceFeatures_<TFeat...>::create(const PhyDeviceBox& phyDeviceBox) noexcept
    -> std::expected<PhyDeviceFeatures_, Error> {
    PhyDeviceFeatures_ props;
    const vk::PhysicalDevice phyDevice = phyDeviceBox.getPhyDevice();

    phyDevice.getFeatures2(props.getPFeature());

    return props;
}

using DefaultPhyDeviceFeatures =
    PhyDeviceFeatures_<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                       vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features>;

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical/features.cpp"
#endif
