#pragma once

#include <expected>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class QueueManager {
    QueueManager(vk::Queue queue) noexcept;

public:
    [[nodiscard]] static std::expected<QueueManager, Error> create(DeviceManager& deviceMgr,
                                                                   vk::QueueFlags type) noexcept;

    [[nodiscard]] vk::Queue getQueue() const noexcept { return queue_; }

private:
    vk::Queue queue_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/queue.cpp"
#endif
