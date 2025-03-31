#pragma once

#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/device/queue_family.hpp"

namespace vkc {

class CommandPoolManager {
public:
    CommandPoolManager(DeviceManager& deviceMgr, const QueueFamilyManager& queueFamilyMgr);
    ~CommandPoolManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getCommandPool(this Self&& self) noexcept {
        return std::forward_like<Self>(self).commandPool_;
    }

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::CommandPool commandPool_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/pool.cpp"
#endif
