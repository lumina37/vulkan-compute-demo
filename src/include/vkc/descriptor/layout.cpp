#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/layout.hpp"
#endif

namespace vkc {

DescSetLayoutManager::DescSetLayoutManager(DeviceManager& deviceMgr,
                                           const std::span<const vk::DescriptorSetLayoutBinding> bindings)
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
