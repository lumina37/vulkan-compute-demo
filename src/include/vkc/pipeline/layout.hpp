#pragma once

#include <expected>
#include <functional>
#include <memory>

#include "vkc/descriptor/layout.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class PipelineLayoutBox {
    PipelineLayoutBox(std::shared_ptr<DeviceBox>&& pDeviceBox, vk::PipelineLayout pipelineLayout) noexcept;

public:
    PipelineLayoutBox(const PipelineLayoutBox&) = delete;
    PipelineLayoutBox(PipelineLayoutBox&& rhs) noexcept;
    ~PipelineLayoutBox() noexcept;

    using TDescSetLayoutBoxCRef = std::reference_wrapper<const DescSetLayoutBox>;

private:
    [[nodiscard]] static std::expected<PipelineLayoutBox, Error> _create(
        std::shared_ptr<DeviceBox>&& pDeviceBox, const std::span<const TDescSetLayoutBoxCRef>& descSetLayoutBoxCRefs,
        const vk::PushConstantRange* pPushConstantRange) noexcept;

public:
    [[nodiscard]] static std::expected<PipelineLayoutBox, Error> create(
        std::shared_ptr<DeviceBox> pDeviceBox, std::span<const TDescSetLayoutBoxCRef> descSetLayoutBoxCRefs) noexcept;

    [[nodiscard]] static std::expected<PipelineLayoutBox, Error> createWithPushConstant(
        std::shared_ptr<DeviceBox> pDeviceBox, std::span<const TDescSetLayoutBoxCRef> descSetLayoutBoxCRefs,
        const vk::PushConstantRange& pushConstantRange) noexcept;

    [[nodiscard]] vk::PipelineLayout getPipelineLayout() const noexcept { return pipelineLayout_; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    vk::PipelineLayout pipelineLayout_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline/layout.cpp"
#endif
