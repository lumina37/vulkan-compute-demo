#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/layout.hpp"
#include "vkc/device/logical.hpp"

namespace vkc {

class PipelineLayoutManager {
public:
    inline PipelineLayoutManager(DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr);
    inline PipelineLayoutManager(DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                                 const vk::PushConstantRange& pushConstantRange);
    inline ~PipelineLayoutManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPipelineLayout(this Self&& self) noexcept {
        return std::forward_like<Self>(self).pipelineLayout_;
    }

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::PipelineLayout pipelineLayout_;
};

inline PipelineLayoutManager::PipelineLayoutManager(DeviceManager& deviceMgr,
                                                    const DescSetLayoutManager& descSetLayoutMgr)
    : deviceMgr_(deviceMgr) {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    pipelineLayoutInfo.setSetLayouts(descSetLayout);

    auto& device = deviceMgr.getDevice();
    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);
}

PipelineLayoutManager::PipelineLayoutManager(DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr,
                                             const vk::PushConstantRange& pushConstantRange)
    : deviceMgr_(deviceMgr) {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    pipelineLayoutInfo.setSetLayouts(descSetLayout);
    pipelineLayoutInfo.setPushConstantRanges(pushConstantRange);

    auto& device = deviceMgr.getDevice();
    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);
}

PipelineLayoutManager::~PipelineLayoutManager() noexcept {
    auto& device = deviceMgr_.getDevice();
    device.destroyPipelineLayout(pipelineLayout_);
}

}  // namespace vkc
