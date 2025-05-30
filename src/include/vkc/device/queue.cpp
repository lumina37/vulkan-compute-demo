#pragma once

#include <expected>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/queue.hpp"
#endif

namespace vkc {

QueueManager::QueueManager(vk::Queue queue) noexcept : queue_(queue) {}

std::expected<QueueManager, Error> QueueManager::create(DeviceManager& deviceMgr, vk::QueueFlags type) noexcept {
    auto queueRes = deviceMgr.getQueue(type);
    if (!queueRes) return std::unexpected{std::move(queueRes.error())};
    const vk::Queue queue = queueRes.value();

    return QueueManager{queue};
}

}  // namespace vkc
