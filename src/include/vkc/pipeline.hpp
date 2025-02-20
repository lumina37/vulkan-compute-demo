#pragma once

#include <array>
#include <cstdint>
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
    inline PipelineManager(DeviceManager& deviceMgr, const PipelineLayoutManager& pipelineLayoutMgr,
                           const ShaderManager& computeShaderMgr);
    inline ~PipelineManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPipeline(this Self&& self) noexcept {
        return std::forward_like<Self>(self).pipeline_;
    }

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::Pipeline pipeline_;
};

PipelineManager::PipelineManager(DeviceManager& deviceMgr, const PipelineLayoutManager& pipelineLayoutMgr,
                                 const ShaderManager& computeShaderMgr)
    : deviceMgr_(deviceMgr) {
    vk::ComputePipelineCreateInfo pipelineInfo;

    // Shaders
    vk::PipelineShaderStageCreateInfo computeShaderStageInfo;
    computeShaderStageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
    computeShaderStageInfo.setModule(computeShaderMgr.getShaderModule());
    computeShaderStageInfo.setPName("main");

    // Group Size
    constexpr std::array specializationEntries{
        vk::SpecializationMapEntry{0, 0 * sizeof(uint32_t), sizeof(uint32_t)},
        vk::SpecializationMapEntry{1, 1 * sizeof(uint32_t), sizeof(uint32_t)},
        vk::SpecializationMapEntry{2, 2 * sizeof(uint32_t), sizeof(uint32_t)},
    };
    constexpr std::array<uint32_t, 3> specializationData{16, 16, 1};

    vk::SpecializationInfo specializationInfo;
    specializationInfo.setMapEntries(specializationEntries);
    specializationInfo.setDataSize(specializationData.size() * sizeof(uint32_t));
    specializationInfo.setPData(specializationData.data());
    computeShaderStageInfo.setPSpecializationInfo(&specializationInfo);

    pipelineInfo.setStage(computeShaderStageInfo);

    // Pipeline Layout
    const auto& pipelineLayout = pipelineLayoutMgr.getPipelineLayout();
    pipelineInfo.setLayout(pipelineLayout);

    // Create Pipeline
    auto& device = deviceMgr.getDevice();
    auto pipelineResult = device.createComputePipeline(nullptr, pipelineInfo);
    if constexpr (ENABLE_DEBUG) {
        if (pipelineResult.result != vk::Result::eSuccess) {
            std::println(std::cerr, "Failed to create graphics pipeline. err: {}", (int)pipelineResult.result);
        }
    }
    pipeline_ = pipelineResult.value;
}

PipelineManager::~PipelineManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyPipeline(pipeline_);
}

}  // namespace vkc
