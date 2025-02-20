#pragma once

#include <vulkan/vulkan.h>

#include "vkc/device/logical.hpp"
#include "vkc/queue_family.hpp"

namespace vkc {

class QueueManager {
public:
    inline QueueManager(DeviceManager& deviceMgr, const QueueFamilyManager& queueFamilyMgr);

    template <typename Self>
    [[nodiscard]] auto&& getComputeQueue(this Self&& self) noexcept {
        return std::forward_like<Self>(self).computeQueue_;
    }

private:
    vk::Queue computeQueue_;
};

QueueManager::QueueManager(DeviceManager& deviceMgr, const QueueFamilyManager& queueFamilyMgr) {
    auto& device = deviceMgr.getDevice();
    computeQueue_ = device.getQueue(queueFamilyMgr.getComputeQFamilyIndex(), 0);
}

}  // namespace vkc
