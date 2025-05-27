#pragma once

#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class FenceManager {
    FenceManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Fence fence) noexcept;

public:
    FenceManager(FenceManager&& rhs) noexcept;
    ~FenceManager() noexcept;

    [[nodiscard]] static std::expected<FenceManager, Error> create(std::shared_ptr<DeviceManager> pDeviceMgr) noexcept;

    [[nodiscard]] vk::Fence getFence() const noexcept { return fence_; }

    [[nodiscard]] std::expected<void, Error> wait() noexcept;
    [[nodiscard]] std::expected<void, Error> reset() noexcept;

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::Fence fence_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/fence.cpp"
#endif
