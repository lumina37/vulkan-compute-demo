#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/device/queue_family.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/queue.hpp"
#endif

namespace vkc {

QueueManager::QueueManager(DeviceManager& deviceMgr, const QueueFamilyManager& queueFamilyMgr) {
    auto& device = deviceMgr.getDevice();
    computeQueue_ = device.getQueue(queueFamilyMgr.getComputeQFamilyIndex(), 0);
}

}  // namespace vkc
