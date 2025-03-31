#include <span>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/layout.hpp"
#endif

namespace vkc {

DescSetLayoutManager::DescSetLayoutManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                                           const std::span<const vk::DescriptorSetLayoutBinding> bindings)
    : pDdeviceMgr_(pDeviceMgr) {
    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.setBindings(bindings);

    auto& device = pDeviceMgr->getDevice();
    descSetlayout_ = device.createDescriptorSetLayout(layoutInfo);
}

DescSetLayoutManager::~DescSetLayoutManager() noexcept {
    auto& device = pDdeviceMgr_->getDevice();
    device.destroyDescriptorSetLayout(descSetlayout_);
}

}  // namespace vkc
