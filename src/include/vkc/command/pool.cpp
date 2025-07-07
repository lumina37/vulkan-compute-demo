#include <cstdint>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/pool.hpp"
#endif

namespace vkc {

CommandPoolBox::CommandPoolBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::CommandPool commandPool,
                               uint32_t queueFamilyIdx) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), commandPool_(commandPool), queueFamilyIdx_(queueFamilyIdx) {}

CommandPoolBox::CommandPoolBox(CommandPoolBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      commandPool_(std::exchange(rhs.commandPool_, nullptr)),
      queueFamilyIdx_(rhs.queueFamilyIdx_) {}

CommandPoolBox::~CommandPoolBox() noexcept {
    if (commandPool_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyCommandPool(commandPool_);
    commandPool_ = nullptr;
}

std::expected<CommandPoolBox, Error> CommandPoolBox::create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                            uint32_t queueFamilyIdx) noexcept {
    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    commandPoolInfo.setQueueFamilyIndex(queueFamilyIdx);

    vk::Device device = pDeviceBox->getDevice();
    const auto [commandPoolRes, commandPool] = device.createCommandPool(commandPoolInfo);
    if (commandPoolRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, commandPoolRes}};
    }

    return CommandPoolBox{std::move(pDeviceBox), commandPool, queueFamilyIdx};
}

}  // namespace vkc
