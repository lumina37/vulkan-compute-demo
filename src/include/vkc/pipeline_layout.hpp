#pragma once

#include <expected>
#include <functional>
#include <memory>

#include "vkc/descriptor/layout.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class PipelineLayoutManager {
    PipelineLayoutManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::PipelineLayout pipelineLayout) noexcept;

public:
    using TDescSetLayoutMgrCRef = std::reference_wrapper<const DescSetLayoutManager>;
    PipelineLayoutManager(PipelineLayoutManager&& rhs) noexcept;
    ~PipelineLayoutManager() noexcept;

    [[nodiscard]] static std::expected<PipelineLayoutManager, Error> create(
        std::shared_ptr<DeviceManager> pDeviceMgr,
        std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs) noexcept;

    [[nodiscard]] static std::expected<PipelineLayoutManager, Error> createWithPushConstant(
        std::shared_ptr<DeviceManager> pDeviceMgr, std::span<const TDescSetLayoutMgrCRef> descSetLayoutMgrCRefs,
        const vk::PushConstantRange& pushConstantRange) noexcept;

    [[nodiscard]] vk::PipelineLayout getPipelineLayout() const noexcept { return pipelineLayout_; }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::PipelineLayout pipelineLayout_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/pipeline_layout.cpp"
#endif
