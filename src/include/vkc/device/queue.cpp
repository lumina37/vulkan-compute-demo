#pragma once

#include <cstdint>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/queue.hpp"
#endif

namespace vkc {

QueueManager::QueueManager(DeviceManager& deviceMgr, const uint32_t queueFamilyIdx) {
    auto& device = deviceMgr.getDevice();
    computeQueue_ = device.getQueue(queueFamilyIdx, 0);
}

}  // namespace vkc
