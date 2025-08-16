#pragma once

#include <cstdint>
#include <expected>
#include <utility>

#include "vkc/device/extensions.hpp"
#include "vkc/device/physical/box.hpp"
#include "vkc/device/physical/concepts.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class DefaultPhyDeviceProps {
public:
    DefaultPhyDeviceProps() noexcept = default;
    DefaultPhyDeviceProps(const DefaultPhyDeviceProps&) = delete;
    DefaultPhyDeviceProps(DefaultPhyDeviceProps&&) noexcept = default;

    [[nodiscard]] static std::expected<DefaultPhyDeviceProps, Error> create(const PhyDeviceBox& phyDeviceBox) noexcept;
    [[nodiscard]] std::expected<float, Error> score() const noexcept;

    [[nodiscard]] vk::PhysicalDeviceFeatures2* getPFeature() noexcept { return &featureChain_.get(); }

    // Members
    ExtEntries_<vk::ExtensionProperties> extensions;
    uint32_t apiVersion;
    vk::PhysicalDeviceType deviceType;
    uint32_t maxSharedMemSize;
    float timestampPeriod;
    bool supportTimeQuery;
    bool supportPerfQuery;
    bool supportFp16;

private:
    vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDevicePerformanceQueryFeaturesKHR,
                       vk::PhysicalDeviceShaderFloat16Int8Features, vk::PhysicalDeviceHostQueryResetFeatures>
        featureChain_;
};

template <CPhyDeviceProps TDProps_>
class PhyDeviceWithProps_ {
public:
    using TDProps = TDProps_;

    PhyDeviceWithProps_(PhyDeviceWithProps_&&) noexcept = default;
    PhyDeviceWithProps_(PhyDeviceBox&& phyDeviceBox, TDProps&& phyDeviceProps) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPhyDeviceBox(this Self&& self) noexcept {
        return std::forward_like<Self>(self).phyDeviceBox_;
    }
    [[nodiscard]] TDProps& getPhyDeviceProps() noexcept { return phyDeviceProps_; }
    [[nodiscard]] const TDProps& getPhyDeviceProps() const noexcept { return phyDeviceProps_; }

private:
    PhyDeviceBox phyDeviceBox_;
    TDProps phyDeviceProps_;
};

template <CPhyDeviceProps TDProps>
PhyDeviceWithProps_<TDProps>::PhyDeviceWithProps_(PhyDeviceBox&& phyDeviceBox, TDProps&& phyDeviceProps) noexcept
    : phyDeviceBox_(std::move(phyDeviceBox)), phyDeviceProps_(std::move(phyDeviceProps)) {}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical/props.cpp"
#endif
