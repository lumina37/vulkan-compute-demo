#include <memory>
#include <ranges>
#include <vector>

#include <vulkan/vulkan.hpp>

#include "vkc/descriptor/layout.hpp"
#include "vkc/device/logical.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline_layout.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

PipelineLayoutManager::PipelineLayoutManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                                             const std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs)
    : pDeviceMgr_(pDeviceMgr) {
    const auto descSetLayouts = descSetLayoutMgrCRefs | rgs::views::transform([](const TDescSetLayoutMgrCRef& mgrRef) {
                                    const auto& descSetLayoutMgr = mgrRef.get();
                                    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
                                    return descSetLayout;
                                }) |
                                rgs::to<std::vector>();

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(descSetLayouts);

    auto& device = pDeviceMgr->getDevice();
    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);
}

PipelineLayoutManager::PipelineLayoutManager(const std::shared_ptr<DeviceManager>& pDeviceMgr,
                                             const std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs,
                                             const vk::PushConstantRange& pushConstantRange)
    : pDeviceMgr_(pDeviceMgr) {
    const auto descSetLayouts = descSetLayoutMgrCRefs | rgs::views::transform([](const TDescSetLayoutMgrCRef& mgrRef) {
                                    const auto& descSetLayoutMgr = mgrRef.get();
                                    const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
                                    return descSetLayout;
                                }) |
                                rgs::to<std::vector>();

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(descSetLayouts);
    pipelineLayoutInfo.setPushConstantRanges(pushConstantRange);

    auto& device = pDeviceMgr->getDevice();
    pipelineLayout_ = device.createPipelineLayout(pipelineLayoutInfo);
}

PipelineLayoutManager::~PipelineLayoutManager() noexcept {
    auto& device = pDeviceMgr_->getDevice();
    device.destroyPipelineLayout(pipelineLayout_);
}

}  // namespace vkc
