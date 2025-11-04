#pragma once

#include <concepts>

#include "vkc/device/physical/box.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"

namespace vkc {

template <typename Self>
concept CPhyDeviceProps = requires(const PhyDeviceBox& phyDeviceBox) {
    // Init from
    { Self::create(phyDeviceBox) } noexcept -> std::same_as<std::expected<Self, Error>>;
} && requires(const Self& self) {
    // Evaluate the priority score
    { self.score() } noexcept -> std::same_as<std::expected<float, Error>>;
} && std::is_move_constructible_v<Self>;

}  // namespace vkc
