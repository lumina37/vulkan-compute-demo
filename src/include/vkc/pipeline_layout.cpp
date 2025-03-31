#include <memory>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/layout.hpp"
#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline_layout.hpp"
#endif

namespace vkc {

PipelineLayoutManager::PipelineLayoutManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                                             const DescSetLayoutManager& descSetLayoutMgr)
    : pDeviceMgr_(pDeviceMgr) {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    pipelineLayoutInfo.setSetLayouts(descSetLayout);

    auto& device = pDeviceMgr->getDevice();
    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);
}

PipelineLayoutManager::PipelineLayoutManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                                             const DescSetLayoutManager& descSetLayoutMgr,
                                             const vk::PushConstantRange& pushConstantRange)
    : pDeviceMgr_(pDeviceMgr) {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    pipelineLayoutInfo.setSetLayouts(descSetLayout);
    pipelineLayoutInfo.setPushConstantRanges(pushConstantRange);

    auto& device = pDeviceMgr->getDevice();
    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);
}

PipelineLayoutManager::~PipelineLayoutManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    device.destroyPipelineLayout(pipelineLayout_);
}

}  // namespace vkc
