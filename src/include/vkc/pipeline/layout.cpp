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
#    include "vkc/pipeline/layout.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

PipelineLayoutBox::PipelineLayoutBox(std::shared_ptr<DeviceBox>&& pDeviceBox,
                                     vk::PipelineLayout pipelineLayout) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), pipelineLayout_(pipelineLayout) {}

PipelineLayoutBox::PipelineLayoutBox(PipelineLayoutBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)), pipelineLayout_(std::exchange(rhs.pipelineLayout_, nullptr)) {}

PipelineLayoutBox::~PipelineLayoutBox() noexcept {
    if (pipelineLayout_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    device.destroyPipelineLayout(pipelineLayout_);
    pipelineLayout_ = nullptr;
}

std::expected<PipelineLayoutBox, Error> PipelineLayoutBox::_create(
    std::shared_ptr<DeviceBox>&& pDeviceBox, const std::span<const TDescSetLayoutBoxCRef>& descSetLayoutBoxCRefs,
    const vk::PushConstantRange* pPushConstantRange) noexcept {
    const auto descSetLayouts =
        descSetLayoutBoxCRefs | rgs::views::transform(DescSetLayoutBox::exposeDescSetLayout) | rgs::to<std::vector>();

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(descSetLayouts);
    if (pPushConstantRange != nullptr) {
        pipelineLayoutInfo.setPPushConstantRanges(pPushConstantRange);
        pipelineLayoutInfo.setPushConstantRangeCount(1);
    }

    vk::Device device = pDeviceBox->getDevice();
    const auto [pipelineLayoutRes, pipelineLayout] = device.createPipelineLayout(pipelineLayoutInfo);
    if (pipelineLayoutRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, pipelineLayoutRes}};
    }

    return PipelineLayoutBox{std::move(pDeviceBox), pipelineLayout};
}

std::expected<PipelineLayoutBox, Error> PipelineLayoutBox::create(
    std::shared_ptr<DeviceBox> pDeviceBox,
    const std::span<const TDescSetLayoutBoxCRef> descSetLayoutBoxCRefs) noexcept {
    return _create(std::move(pDeviceBox), descSetLayoutBoxCRefs, nullptr);
}

std::expected<PipelineLayoutBox, Error> PipelineLayoutBox::createWithPushConstant(
    std::shared_ptr<DeviceBox> pDeviceBox, const std::span<const TDescSetLayoutBoxCRef> descSetLayoutBoxCRefs,
    const vk::PushConstantRange& pushConstantRange) noexcept {
    return _create(std::move(pDeviceBox), descSetLayoutBoxCRefs, &pushConstantRange);
}

}  // namespace vkc
