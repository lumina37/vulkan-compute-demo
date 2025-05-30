#pragma once

#include <expected>
#include <memory>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/shader.hpp"

namespace vkc {

class PipelineManager {
    PipelineManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Pipeline pipeline,
                    vk::PipelineBindPoint bindPoint) noexcept;

public:
    PipelineManager(PipelineManager&& rhs) noexcept;
    ~PipelineManager() noexcept;

    [[nodiscard]] static std::expected<PipelineManager, Error> createCompute(
        std::shared_ptr<DeviceManager> pDeviceMgr, const PipelineLayoutManager& pipelineLayoutMgr,
        const ShaderManager& shaderMgr, const vk::SpecializationInfo& specInfo) noexcept;

    [[nodiscard]] vk::Pipeline getPipeline() const noexcept { return pipeline_; }
    [[nodiscard]] vk::PipelineBindPoint getBindPoint() const noexcept { return bindPoint_; }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::Pipeline pipeline_;
    vk::PipelineBindPoint bindPoint_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline.cpp"
#endif
