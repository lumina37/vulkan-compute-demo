#pragma once

#include <cstdint>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

namespace vkc {

class CommandPoolManager {
public:
    CommandPoolManager(DeviceManager& deviceMgr, uint32_t queueFamilyIdx);
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
