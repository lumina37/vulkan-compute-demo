#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/layout.hpp"
#include "vkc/device/logical.hpp"

namespace vkc {

class PipelineLayoutManager {
public:
    inline PipelineLayoutManager(const DeviceManager& deviceMgr, const DescSetLayoutManager& descSetLayoutMgr);
    inline ~PipelineLayoutManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getPipelineLayout(this Self& self) noexcept {
        return std::forward_like<Self>(self).pipelineLayout_;
    }

private:
    const DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::PipelineLayout pipelineLayout_;
};

PipelineLayoutManager::PipelineLayoutManager(const DeviceManager& deviceMgr,
                                             const DescSetLayoutManager& descSetLayoutMgr)
    : deviceMgr_(deviceMgr) {
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
    pipelineLayoutInfo.setSetLayouts(descSetLayout);

    const auto& device = deviceMgr.getDevice();
    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);
}

PipelineLayoutManager::~PipelineLayoutManager() noexcept {
    const auto& device = deviceMgr_.getDevice();
    device.destroyPipelineLayout(pipelineLayout_);
}

}  // namespace vkc
