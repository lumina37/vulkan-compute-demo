#pragma once

#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/binding.hpp"
#include "vkc/device/logical.hpp"

namespace vkc {

class DescSetLayoutManager {
public:
    inline DescSetLayoutManager(DeviceManager& deviceMgr, const std::span<vk::DescriptorSetLayoutBinding> bindings);
    inline ~DescSetLayoutManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescSetLayout(this Self&& self) noexcept {
        return std::forward_like<Self>(self).descSetlayout_;
    }

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::DescriptorSetLayout descSetlayout_;
};

DescSetLayoutManager::DescSetLayoutManager(DeviceManager& deviceMgr,
                                           const std::span<vk::DescriptorSetLayoutBinding> bindings)
    : deviceMgr_(deviceMgr) {
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.setBindings(bindings);

    auto& device = deviceMgr.getDevice();
    descSetlayout_ = device.createDescriptorSetLayout(layoutInfo);
}

DescSetLayoutManager::~DescSetLayoutManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyDescriptorSetLayout(descSetlayout_);
}

}  // namespace vkc
