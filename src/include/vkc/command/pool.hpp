#pragma once

#include <cstdint>
#include <expected>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class CommandPoolBox {
    CommandPoolBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::CommandPool commandPool,
                   uint32_t queueFamilyIdx) noexcept;

public:
    CommandPoolBox(const CommandPoolBox&) = delete;
    CommandPoolBox(CommandPoolBox&& rhs) noexcept;
    ~CommandPoolBox() noexcept;

    [[nodiscard]] static std::expected<CommandPoolBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                     uint32_t queueFamilyIdx) noexcept;

    [[nodiscard]] vk::CommandPool getCommandPool() const noexcept { return commandPool_; }
    [[nodiscard]] uint32_t getQueueFamilyIdx() const noexcept { return queueFamilyIdx_; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::CommandPool commandPool_;
    uint32_t queueFamilyIdx_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/pool.cpp"
#endif
