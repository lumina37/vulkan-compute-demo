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
    vk::Device device = pDeviceMgr_->getDevice();
    device.destroyPipelineLayout(pipelineLayout_);
    pipelineLayout_ = nullptr;
}

std::expected<PipelineLayoutManager, Error> PipelineLayoutManager::_create(
    std::shared_ptr<DeviceManager>&& pDeviceMgr, const std::span<const TDescSetLayoutMgrCRef>& descSetLayoutMgrCRefs,
    const vk::PushConstantRange* pPushConstantRange) noexcept {
    const auto descSetLayouts = descSetLayoutMgrCRefs |
                                rgs::views::transform(DescSetLayoutManager::exposeDescSetLayout) |
                                rgs::to<std::vector>();

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(descSetLayouts);
    if (pPushConstantRange != nullptr) {
        pipelineLayoutInfo.setPPushConstantRanges(pPushConstantRange);
        pipelineLayoutInfo.setPushConstantRangeCount(1);
    }

    vk::Device device = pDeviceMgr->getDevice();
    const auto [pipelineLayoutRes, pipelineLayout] = device.createPipelineLayout(pipelineLayoutInfo);
    if (pipelineLayoutRes != vk::Result::eSuccess) {
        return std::unexpected{Error{pipelineLayoutRes}};
    }

    return PipelineLayoutManager{std::move(pDeviceMgr), pipelineLayout};
}

std::expected<PipelineLayoutManager, Error> PipelineLayoutManager::create(
    std::shared_ptr<DeviceManager> pDeviceMgr,
    const std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs) noexcept {
    return _create(std::move(pDeviceMgr), descSetLayoutMgrCRefs, nullptr);
}

std::expected<PipelineLayoutManager, Error> PipelineLayoutManager::createWithPushConstant(
    std::shared_ptr<DeviceManager> pDeviceMgr, const std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs,
    const vk::PushConstantRange& pushConstantRange) noexcept {
    return _create(std::move(pDeviceMgr), descSetLayoutMgrCRefs, &pushConstantRange);
}

}  // namespace vkc
