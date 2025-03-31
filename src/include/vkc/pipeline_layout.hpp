#pragma once

#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/layout.hpp"
#include "vkc/device/logical.hpp"

namespace vkc {

class PipelineLayoutManager {
public:
    PipelineLayoutManager(const std::shared_ptr<DeviceManager>& pDeviceMgr, const DescSetLayoutManager& descSetLayoutMgr);
    PipelineLayoutManager(const std::shared_ptr<DeviceManager>& pDeviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                          const vk::PushConstantRange& pushConstantRange);
    ~PipelineLayoutManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPipelineLayout(this Self&& self) noexcept {
        return std::forward_like<Self>(self).pipelineLayout_;
    }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::PipelineLayout pipelineLayout_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline_layout.cpp"
#endif
