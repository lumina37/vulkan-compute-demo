#include <expected>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

#include "vkc/descriptor/layout.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline_layout.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

PipelineLayoutManager::PipelineLayoutManager(std::shared_ptr<DeviceManager>&& pDeviceMgr,
                                             vk::PipelineLayout pipelineLayout) noexcept
    : pDeviceMgr_(std::move(pDeviceMgr)), pipelineLayout_(pipelineLayout) {}

PipelineLayoutManager::PipelineLayoutManager(PipelineLayoutManager&& rhs) noexcept
    : pDeviceMgr_(std::move(rhs.pDeviceMgr_)), pipelineLayout_(std::exchange(rhs.pipelineLayout_, nullptr)) {}

PipelineLayoutManager::~PipelineLayoutManager() noexcept {
    if (pipelineLayout_ == nullptr) return;
    auto& device = pDeviceMgr_->getDevice();
    device.destroyPipelineLayout(pipelineLayout_);
    pipelineLayout_ = nullptr;
}

std::expected<PipelineLayoutManager, Error> PipelineLayoutManager::create(
    std::shared_ptr<DeviceManager> pDeviceMgr, std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs) noexcept {
    const auto genDescSetLayout = [](const TDescSetLayoutMgrCRef& mgrRef) {
        const auto& descSetLayoutMgr = mgrRef.get();
        const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
        return descSetLayout;
    };

    const auto descSetLayouts =
        descSetLayoutMgrCRefs | rgs::views::transform(genDescSetLayout) | rgs::to<std::vector>();

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(descSetLayouts);

    auto& device = pDeviceMgr->getDevice();
    const auto [pipelineLayoutRes, pipelineLayout] = device.createPipelineLayout(pipelineLayoutInfo);
    if (pipelineLayoutRes != vk::Result::eSuccess) {
        return std::unexpected{Error{pipelineLayoutRes}};
    }

    return PipelineLayoutManager{std::move(pDeviceMgr), pipelineLayout};
}

std::expected<PipelineLayoutManager, Error> PipelineLayoutManager::createWithPushConstant(
    std::shared_ptr<DeviceManager> pDeviceMgr, std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs,
    const vk::PushConstantRange& pushConstantRange) noexcept {
    const auto genDescSetLayout = [](const TDescSetLayoutMgrCRef& mgrRef) {
        const auto& descSetLayoutMgr = mgrRef.get();
        const auto& descSetLayout = descSetLayoutMgr.getDescSetLayout();
        return descSetLayout;
    };

    const auto descSetLayouts =
        descSetLayoutMgrCRefs | rgs::views::transform(genDescSetLayout) | rgs::to<std::vector>();

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(descSetLayouts);
    pipelineLayoutInfo.setPushConstantRanges(pushConstantRange);

    auto& device = pDeviceMgr->getDevice();
    const auto [pipelineLayoutRes, pipelineLayout] = device.createPipelineLayout(pipelineLayoutInfo);
    if (pipelineLayoutRes != vk::Result::eSuccess) {
        return std::unexpected{Error{pipelineLayoutRes}};
    }

    return PipelineLayoutManager{std::move(pDeviceMgr), pipelineLayout};
}

}  // namespace vkc
