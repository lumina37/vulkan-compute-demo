#pragma once

#include <cstdint>
#include <expected>
#include <limits>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class FenceBox {
    FenceBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Fence fence) noexcept;

public:
    FenceBox(const FenceBox&) = delete;
    FenceBox(FenceBox&& rhs) noexcept;
    ~FenceBox() noexcept;

    [[nodiscard]] static std::expected<FenceBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox) noexcept;

    [[nodiscard]] vk::Fence getFence() const noexcept { return fence_; }

    [[nodiscard]] std::expected<void, Error> wait(uint64_t timeout = std::numeric_limits<uint64_t>::max()) noexcept;
    [[nodiscard]] std::expected<void, Error> reset() noexcept;

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::Fence fence_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/sync/fence.cpp"
#endif
