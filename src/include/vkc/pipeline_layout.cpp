#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/layout.hpp"
#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline_layout.hpp"
#endif

namespace vkc {

PipelineLayoutManager::PipelineLayoutManager(DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr)
    : deviceMgr_(deviceMgr) {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    pipelineLayoutInfo.setSetLayouts(descSetLayout);

    auto& device = deviceMgr.getDevice();
    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);
}

PipelineLayoutManager::PipelineLayoutManager(DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                                             const vk::PushConstantRange& pushConstantRange)
    : deviceMgr_(deviceMgr) {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    pipelineLayoutInfo.setSetLayouts(descSetLayout);
    pipelineLayoutInfo.setPushConstantRanges(pushConstantRange);

    auto& device = deviceMgr.getDevice();
    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);
}

PipelineLayoutManager::~PipelineLayoutManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyPipelineLayout(pipelineLayout_);
}

}  // namespace vkc
