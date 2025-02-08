#pragma once

#include <array>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

namespace vkc {

class DescSetLayoutManager {
public:
    inline DescSetLayoutManager(const DeviceManager& deviceMgr);
    inline ~DescSetLayoutManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getDescSetLayout(this Self& self) noexcept {
        return std::forward_like<Self>(self).descSetlayout_;
    }

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::DescriptorSetLayout descSetlayout_;
};

DescSetLayoutManager::DescSetLayoutManager(const DeviceManager& deviceMgr) : deviceMgr_(deviceMgr) {
    vk::DescriptorSetLayoutBinding srcBinding;
    srcBinding.setBinding(0);
    srcBinding.setDescriptorCount(1);
    srcBinding.setDescriptorType(vk::DescriptorType::eStorageImage);
    srcBinding.setStageFlags(vk::ShaderStageFlagBits::eCompute);

    vk::DescriptorSetLayoutBinding dstBinding;
    dstBinding.setBinding(1);
    dstBinding.setDescriptorCount(1);
    dstBinding.setDescriptorType(vk::DescriptorType::eStorageImage);
    dstBinding.setStageFlags(vk::ShaderStageFlagBits::eCompute);

    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    const std::array bindings{srcBinding, dstBinding};
    layoutInfo.setBindings(bindings);

    const auto& device = deviceMgr.getDevice();
    descSetlayout_ = device.createDescriptorSetLayout(layoutInfo);
}

DescSetLayoutManager::~DescSetLayoutManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyDescriptorSetLayout(descSetlayout_);
}

}  // namespace vkc
