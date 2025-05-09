#pragma once

#include <concepts>
#include <expected>

#include "vkc/device/physical/manager.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

template <typename Self>
concept CPhyDeviceProps = requires(const PhyDeviceManager& phyDeviceMgr) {
    // Init from
    { Self::create(phyDeviceMgr) } noexcept -> std::same_as<std::expected<Self, Error>>;
} && requires(const Self& self) {
    // Evaluate the priority score
    { self.score() } noexcept -> std::same_as<std::expected<float, Error>>;
} && std::is_move_constructible_v<Self>;

}  // namespace vkc
