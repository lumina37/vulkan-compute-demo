#pragma once

#include <cstdint>
#include <expected>
#include <utility>

#include "vkc/device/concepts.hpp"
#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class PhyDeviceProps {
public:
    PhyDeviceProps() noexcept = default;
    PhyDeviceProps(const PhyDeviceProps&) = delete;
    PhyDeviceProps(PhyDeviceProps&&) noexcept = default;

    [[nodiscard]] static std::expected<PhyDeviceProps, Error> create(const PhyDeviceManager& phyDeviceMgr) noexcept;

    // Members
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
#    include "vkc/device/props.cpp"
#endif
