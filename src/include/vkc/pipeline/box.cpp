#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/pipeline/layout.hpp"
#include "vkc/shader.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline/box.hpp"
#endif

namespace vkc {

PipelineBox::PipelineBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::Pipeline pipeline,
                         vk::PipelineBindPoint bindPoint) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), pipeline_(pipeline), bindPoint_(bindPoint) {}

PipelineBox::PipelineBox(PipelineBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      pipeline_(std::exchange(rhs.pipeline_, nullptr)),
      bindPoint_(rhs.bindPoint_) {}

PipelineBox::~PipelineBox() noexcept {
    if (pipeline_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyPipeline(pipeline_);
    pipeline_ = nullptr;
}

std::expected<PipelineBox, Error> PipelineBox::createCompute(std::shared_ptr<DeviceBox> pDeviceBox,
                                                             const PipelineLayoutBox& pipelineLayoutBox,
                                                             const ShaderBox& shaderBox,
                                                             const vk::SpecializationInfo& specInfo) noexcept {
    vk::ComputePipelineCreateInfo pipelineInfo;

    // Shaders
    vk::PipelineShaderStageCreateInfo shaderStageInfo;
    shaderStageInfo.setStage(vk::ShaderStageFlagBits::eCompute);
    shaderStageInfo.setModule(shaderBox.getShaderModule());
    shaderStageInfo.setPName("main");
    shaderStageInfo.setPSpecializationInfo(&specInfo);
    pipelineInfo.setStage(shaderStageInfo);

    // Pipeline Layout
    vk::PipelineLayout pipelineLayout = pipelineLayoutBox.getPipelineLayout();
    pipelineInfo.setLayout(pipelineLayout);

    // Create Pipeline
    vk::Device device = pDeviceBox->getDevice();
    const auto pipelineRes = device.createComputePipeline(nullptr, pipelineInfo);
    if (pipelineRes.result != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, pipelineRes.result}};
    }
    vk::Pipeline pipeline = pipelineRes.value;

    return PipelineBox{std::move(pDeviceBox), pipeline, vk::PipelineBindPoint::eCompute};
}

}  // namespace vkc
