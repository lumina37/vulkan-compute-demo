#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/layout.hpp"
#include "vkc/descriptor/pool.hpp"
#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/set.hpp"
#endif

namespace vkc {

DescSetManager::DescSetManager(DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                               DescPoolManager& descPoolMgr)
    : deviceMgr_(deviceMgr), descPoolMgr_(descPoolMgr) {
    vk::DescriptorSetAllocateInfo descSetAllocInfo;
    descSetAllocInfo.setDescriptorPool(descPoolMgr.getDescPool());
    descSetAllocInfo.setDescriptorSetCount(1);
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    descSetAllocInfo.setSetLayouts(descSetLayout);

    auto& device = deviceMgr.getDevice();
    const auto& descSets = device.allocateDescriptorSets(descSetAllocInfo);
    descSet_ = descSets[0];
}

DescSetManager::~DescSetManager() noexcept {
    // TODO: maybe free sth. here
}

}  // namespace vkc
