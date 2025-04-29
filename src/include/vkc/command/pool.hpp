#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class CommandPoolManager {
    CommandPoolManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::CommandPool commandPool,
                       uint32_t queueFamilyIdx) noexcept;

public:
    CommandPoolManager(CommandPoolManager&& rhs) noexcept;
    ~CommandPoolManager() noexcept;

    [[nodiscard]] static std::expected<CommandPoolManager, Error> create(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                         uint32_t queueFamilyIdx) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getCommandPool(this Self&& self) noexcept {
        return std::forward_like<Self>(self).commandPool_;
    }

    [[nodiscard]] uint32_t getQueueFamilyIdx() const noexcept { return queueFamilyIdx_; }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::CommandPool commandPool_;
    uint32_t queueFamilyIdx_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/pool.cpp"
#endif
