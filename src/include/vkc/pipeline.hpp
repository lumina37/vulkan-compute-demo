#pragma once

#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/shader.hpp"

namespace vkc {

class PipelineManager {
public:
    PipelineManager(const std::shared_ptr<DeviceManager>& pDeviceMgr, const PipelineLayoutManager& pipelineLayoutMgr,
                    const ShaderManager& computeShaderMgr);
    ~PipelineManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPipeline(this Self&& self) noexcept {
        return std::forward_like<Self>(self).pipeline_;
    }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::Pipeline pipeline_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline.cpp"
#endif
