#pragma once

#include <iostream>
#include <print>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/shader.hpp"

namespace vkc {

class PipelineManager {
public:
    inline PipelineManager(const DeviceManager& deviceMgr, const PipelineLayoutManager& pipelineLayoutMgr,
                           const ShaderManager& computeShaderMgr);
    inline ~PipelineManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPipeline(this Self& self) noexcept {
        return std::forward_like<Self>(self).pipeline_;
    }

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::Pipeline pipeline_;
};

PipelineManager::PipelineManager(const DeviceManager& deviceMgr, const PipelineLayoutManager& pipelineLayoutMgr,
                                 const ShaderManager& computeShaderMgr)
    : deviceMgr_(deviceMgr) {
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
    const auto& device = deviceMgr.getDevice();
    auto pipelineResult = device.createComputePipeline(nullptr, pipelineInfo);
    if constexpr (ENABLE_DEBUG) {
        if (pipelineResult.result != vk::Result::eSuccess) {
            std::println(std::cerr, "Failed to create graphics pipeline. err: {}", (int)pipelineResult.result);
        }
    }
    pipeline_ = pipelineResult.value;
}

PipelineManager::~PipelineManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyPipeline(pipeline_);
}

}  // namespace vkc
