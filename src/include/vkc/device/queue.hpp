#pragma once

#include <cstdint>
#include <expected>
#include <utility>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class QueueManager {
    QueueManager(vk::Queue queue) noexcept;

public:
    [[nodiscard]] static std::expected<QueueManager, Error> create(DeviceManager& deviceMgr,
                                                                   vk::QueueFlags type) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getComputeQueue(this Self&& self) noexcept {
        return std::forward_like<Self>(self).computeQueue_;
    }

private:
    vk::Queue computeQueue_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/queue.cpp"
#endif
