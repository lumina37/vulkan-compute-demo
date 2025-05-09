#pragma once

#include <cstdint>
#include <expected>
#include <utility>

#include "vkc/device/extensions.hpp"
#include "vkc/device/physical/concepts.hpp"
#include "vkc/device/physical/manager.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class DefaultPhyDeviceProps {
public:
    DefaultPhyDeviceProps() noexcept = default;
    DefaultPhyDeviceProps(const DefaultPhyDeviceProps&) = delete;
    DefaultPhyDeviceProps(DefaultPhyDeviceProps&&) noexcept = default;

    [[nodiscard]] static std::expected<DefaultPhyDeviceProps, Error> create(
        const PhyDeviceManager& phyDeviceMgr) noexcept;
    [[nodiscard]] std::expected<float, Error> score() const noexcept;

    // Members
    ExtEntries_<vk::ExtensionProperties> extensions;
    uint32_t apiVersion;
    vk::PhysicalDeviceType deviceType;
    uint32_t maxSharedMemSize;
    float timestampPeriod;
    bool supportFp16;
    bool supportTimeQueryForAllQueue;
};

template <CPhyDeviceProps TDProps_>
class PhyDeviceWithProps_ {
public:
    using TDProps = TDProps_;

    PhyDeviceWithProps_(PhyDeviceManager&& phyDeviceMgr, TDProps&& phyDeviceProps) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPhyDeviceMgr(this Self&& self) noexcept {
        return std::forward_like<Self>(self).phyDeviceMgr_;
    }
    [[nodiscard]] const TDProps& getPhyDeviceProps() const noexcept { return phyDeviceProps_; }

private:
    PhyDeviceManager phyDeviceMgr_;
    TDProps phyDeviceProps_;
};

template <CPhyDeviceProps TDProps>
PhyDeviceWithProps_<TDProps>::PhyDeviceWithProps_(PhyDeviceManager&& phyDeviceMgr, TDProps&& phyDeviceProps) noexcept
    : phyDeviceMgr_(std::move(phyDeviceMgr)), phyDeviceProps_(std::move(phyDeviceProps)) {}

}  // namespace vkc

#ifdef _vkc_LIB_HEADER_ONLY
#    include "vkc/device/physical/props.cpp"
#endif
