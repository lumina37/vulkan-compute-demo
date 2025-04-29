#include <expected>
#include <iostream>
#include <memory>
#include <print>

#include "vkc/helper/vulkan.hpp"

#include "vkc/device/logical.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/shader.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline.hpp"
#endif

namespace vkc {

PipelineManager::PipelineManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Pipeline pipeline) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), pipeline_(pipeline) {}

PipelineManager::PipelineManager(PipelineManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)), pipeline_(std::exchange(rhs.pipeline_, nullptr)) {}

PipelineManager::~PipelineManager() noexcept {
    if (pipeline_ == nullptr) return;
    auto& device = pDeviceMgr_->getDevice();
    device.destroyPipeline(pipeline_);
    pipeline_ = nullptr;
}

std::expected<PipelineManager, Error> PipelineManager::create(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                              const PipelineLayoutManager& pipelineLayoutMgr,
                                                              const ShaderManager& computeShaderMgr,
                                                              const vk::SpecializationInfo& specInfo) noexcept {
    vk::ComputePipelineCreateInfo pipelineInfo;

    // Shaders
    vk::PipelineShaderStageCreateInfo computeShaderStageInfo;
    computeShaderStageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
    computeShaderStageInfo.setModule(computeShaderMgr.getShaderModule());
    computeShaderStageInfo.setPName("main");
    computeShaderStageInfo.setPSpecializationInfo(&specInfo);
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
    vk::Pipeline pipeline = pipelineResult.value;

    return PipelineManager{std::move(pDeviceMgr), pipeline};
}

}  // namespace vkc
