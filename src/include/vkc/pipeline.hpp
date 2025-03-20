#pragma once

#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/shader.hpp"

namespace vkc {

class PipelineManager {
public:
    PipelineManager(DeviceManager& deviceMgr, const PipelineLayoutManager& pipelineLayoutMgr,
                    const ShaderManager& computeShaderMgr);
    ~PipelineManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPipeline(this Self&& self) noexcept {
        return std::forward_like<Self>(self).pipeline_;
    }

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::Pipeline pipeline_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline.cpp"
#endif
