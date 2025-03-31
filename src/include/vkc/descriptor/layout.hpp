#pragma once

#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

namespace vkc {

class DescSetLayoutManager {
public:
    DescSetLayoutManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                         std::span<const vk::DescriptorSetLayoutBinding> bindings);
    ~DescSetLayoutManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescSetLayout(this Self&& self) noexcept {
        return std::forward_like<Self>(self).descSetlayout_;
    }

private:
    std::shared_ptr<DeviceManager> pDdeviceMgr_;

    vk::DescriptorSetLayout descSetlayout_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/layout.cpp"
#endif
