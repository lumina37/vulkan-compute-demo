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
    PipelineManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::Pipeline pipeline) noexcept;

public:
    PipelineManager(PipelineManager&& rhs) noexcept;
    ~PipelineManager() noexcept;

    [[nodiscard]] static std::expected<PipelineManager, Error> create(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                      const PipelineLayoutManager& pipelineLayoutMgr,
                                                                      const ShaderManager& computeShaderMgr,
                                                                      const vk::SpecializationInfo& specInfo) noexcept;

    [[nodiscard]] vk::Pipeline getPipeline() const noexcept { return pipeline_; }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::Pipeline pipeline_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline.cpp"
#endif
