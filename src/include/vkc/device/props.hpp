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

    [[nodiscard]] static std::expected<PhyDeviceProps, Error> create(
        const PhyDeviceManager& phyDeviceMgr) noexcept;

    // Members
    vk::PhysicalDeviceType deviceType;
    uint32_t maxSharedMemSize;
    float timestampPeriod;
    bool supportTimeQueryForAllQueue;
};

template <CPhyDeviceProps TProps_>
class PhyDeviceWithProps_ {
public:
    using TProps = TProps_;

    PhyDeviceWithProps_(PhyDeviceManager&& phyDeviceMgr, TProps&& props) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPhyDeviceMgr(this Self&& self) noexcept {
        return std::forward_like<Self>(self).phyDeviceMgr_;
    }
    [[nodiscard]] const TProps& getProps() const noexcept { return props_; }

private:
    PhyDeviceManager phyDeviceMgr_;
    TProps props_;
};

template <CPhyDeviceProps TProps>
PhyDeviceWithProps_<TProps>::PhyDeviceWithProps_(PhyDeviceManager&& phyDeviceMgr, TProps&& props) noexcept
    : phyDeviceMgr_(std::move(phyDeviceMgr)), props_(std::move(props)) {}

}  // namespace vkc

#ifdef _vkc_LIB_HEADER_ONLY
#    include "vkc/device/props.cpp"
#endif
