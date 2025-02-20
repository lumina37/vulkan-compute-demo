#pragma once

#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/binding.hpp"
#include "vkc/device/logical.hpp"

namespace vkc {

class DescSetLayoutManager {
public:
    inline DescSetLayoutManager(const DeviceManager& deviceMgr,
                                const std::span<vk::DescriptorSetLayoutBinding> bindings);
    inline ~DescSetLayoutManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescSetLayout(this Self& self) noexcept {
        return std::forward_like<Self>(self).descSetlayout_;
    }

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::DescriptorSetLayout descSetlayout_;
};

DescSetLayoutManager::DescSetLayoutManager(const DeviceManager& deviceMgr,
                                           const std::span<vk::DescriptorSetLayoutBinding> bindings)
    : deviceMgr_(deviceMgr) {
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.setBindings(bindings);

    const auto& device = deviceMgr.getDevice();
    descSetlayout_ = device.createDescriptorSetLayout(layoutInfo);
}

DescSetLayoutManager::~DescSetLayoutManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyDescriptorSetLayout(descSetlayout_);
}

}  // namespace vkc
