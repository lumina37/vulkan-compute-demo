#include <iostream>
#include <memory>
#include <print>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/shader.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline.hpp"
#endif

namespace vkc {

PipelineManager::PipelineManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                                 const PipelineLayoutManager& pipelineLayoutMgr, const ShaderManager& computeShaderMgr)
    : pDeviceMgr_(pDeviceMgr) {
    vk::ComputePipelineCreateInfo pipelineInfo;

    // Shaders
    vk::PipelineShaderStageCreateInfo computeShaderStageInfo;
    computeShaderStageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
    computeShaderStageInfo.setModule(computeShaderMgr.getShaderModule());
    computeShaderStageInfo.setPName("main");
    pipelineInfo.setStage(computeShaderStageInfo);

    // Pipeline Layout
    const auto& pipelineLayout = pipelineLayoutMgr.getPipelineLayout();
    pipelineInfo.setLayout(pipelineLayout);

    // Create Pipeline
    auto& device = pDeviceMgr->getDevice();
    auto pipelineResult = device.createComputePipeline(nullptr, pipelineInfo);
    if constexpr (ENABLE_DEBUG) {
        if (pipelineResult.result != vk::Result::eSuccess) {
            std::println(std::cerr, "Failed to create graphics pipeline. err: {}", (int)pipelineResult.result);
        }
    }
    pipeline_ = pipelineResult.value;
}

PipelineManager::~PipelineManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    device.destroyPipeline(pipeline_);
}

}  // namespace vkc
