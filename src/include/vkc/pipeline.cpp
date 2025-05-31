#include <expected>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/shader.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline.hpp"
#endif

namespace vkc {

PipelineManager::PipelineManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Pipeline pipeline,
                                 vk::PipelineBindPoint bindPoint) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), pipeline_(pipeline), bindPoint_(bindPoint) {}

PipelineManager::PipelineManager(PipelineManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)),
      pipeline_(std::exchange(rhs.pipeline_, nullptr)),
      bindPoint_(rhs.bindPoint_) {}

PipelineManager::~PipelineManager() noexcept {
    if (pipeline_ == nullptr) return;
    vk::Device device = pDeviceMgr_->getDevice();
    device.destroyPipeline(pipeline_);
    pipeline_ = nullptr;
}

std::expected<PipelineManager, Error> PipelineManager::createCompute(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                     const PipelineLayoutManager& pipelineLayoutMgr,
                                                                     const ShaderManager& shaderMgr,
                                                                     const vk::SpecializationInfo& specInfo) noexcept {
    vk::ComputePipelineCreateInfo pipelineInfo;

    // Shaders
    vk::PipelineShaderStageCreateInfo shaderStageInfo;
    shaderStageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
    shaderStageInfo.setModule(shaderMgr.getShaderModule());
    shaderStageInfo.setPName("main");
    shaderStageInfo.setPSpecializationInfo(&specInfo);
    pipelineInfo.setStage(shaderStageInfo);

    // Pipeline Layout
    const auto& pipelineLayout = pipelineLayoutMgr.getPipelineLayout();
    pipelineInfo.setLayout(pipelineLayout);

    // Create Pipeline
    vk::Device device = pDeviceMgr->getDevice();
    auto pipelineResult = device.createComputePipeline(nullptr, pipelineInfo);
    if (pipelineResult.result != vk::Result::eSuccess) {
        return std::unexpected{Error{(int)pipelineResult.result, "failed to create compute pipeline"}};
    }
    vk::Pipeline pipeline = pipelineResult.value;

    return PipelineManager{std::move(pDeviceMgr), pipeline, vk::PipelineBindPoint::eCompute};
}

}  // namespace vkc
