#pragma once

#include <concepts>
#include <expected>

#include "vkc/device/physical.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

template <typename Self>
concept CPhyDeviceProps = std::is_move_constructible_v<Self> && requires(const PhyDeviceManager& phyDeviceMgr) {
    // Init from
    { Self::create(phyDeviceMgr) } noexcept -> std::same_as<std::expected<Self, Error>>;
};

}  // namespace vkc
