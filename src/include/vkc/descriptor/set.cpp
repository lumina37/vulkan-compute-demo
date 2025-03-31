#include <memory>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/layout.hpp"
#include "vkc/descriptor/pool.hpp"
#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/descriptor/set.hpp"
#endif

namespace vkc {

DescSetManager::DescSetManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                               const DescSetLayoutManager& descSetLayoutMgr, DescPoolManager& descPoolMgr)
    : pDeviceMgr_(pDeviceMgr) {
    vk::DescriptorSetAllocateInfo descSetAllocInfo;
    auto& descPool = descPoolMgr.getDescPool();
    descSetAllocInfo.setDescriptorPool(descPool);
    descSetAllocInfo.setDescriptorSetCount(1);
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    descSetAllocInfo.setSetLayouts(descSetLayout);

    auto& device = pDeviceMgr->getDevice();
    const auto& descSets = device.allocateDescriptorSets(descSetAllocInfo);
    descSet_ = descSets[0];
}

}  // namespace vkc
