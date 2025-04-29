#pragma once

#include <cstdint>
#include <expected>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/queue.hpp"
#endif

namespace vkc {

QueueManager::QueueManager(vk::Queue queue) noexcept : computeQueue_(queue) {}

std::expected<QueueManager, Error> QueueManager::create(DeviceManager& deviceMgr, uint32_t queueFamilyIdx) noexcept {
    auto& device = deviceMgr.getDevice();
    const vk::Queue computeQueue = device.getQueue(queueFamilyIdx, 0);

    return QueueManager{computeQueue};
}

}  // namespace vkc
