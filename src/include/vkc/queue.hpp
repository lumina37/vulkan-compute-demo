#pragma once

#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/device/queue_family.hpp"

namespace vkc {

class QueueManager {
public:
    QueueManager(DeviceManager& deviceMgr, const QueueFamilyManager& queueFamilyMgr);

    template <typename Self>
    [[nodiscard]] auto&& getComputeQueue(this Self&& self) noexcept {
        return std::forward_like<Self>(self).computeQueue_;
    }

private:
    vk::Queue computeQueue_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/queue.cpp"
#endif
