#pragma once

#include <expected>
#include <memory>
#include <utility>

#include "vkc/command/concepts.hpp"
#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/device/queue.hpp"
#include "vkc/extent.hpp"
#include "vkc/fence.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/query_pool.hpp"
#include "vkc/resource.hpp"

namespace vkc {

struct BlockSize {
    using Tv = uint32_t;
    Tv x, y, z;
};

class CommandBufferManager {
    CommandBufferManager(std::shared_ptr<DeviceManager>&& pDeviceMgr,
                         std::shared_ptr<CommandPoolManager>&& pCommandPoolMgr,
                         vk::CommandBuffer commandBuffer) noexcept;

public:
    CommandBufferManager(CommandBufferManager&& rhs) noexcept;
    ~CommandBufferManager() noexcept;

    [[nodiscard]] static std::expected<CommandBufferManager, Error> create(
        std::shared_ptr<DeviceManager> pDeviceMgr, std::shared_ptr<CommandPoolManager> pCommandPoolMgr) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getCommandBuffer(this Self&& self) noexcept {
        return std::forward_like<Self>(self).commandBuffer_;
    }

    void bindPipeline(PipelineManager& pipelineMgr) noexcept;
    void bindDescSets(DescSetsManager& descSetsMgr, const PipelineLayoutManager& pipelineLayoutMgr) noexcept;

    template <typename TPc>
    void pushConstant(const PushConstantManager<TPc>& pushConstantMgr,
                      const PipelineLayoutManager& pipelineLayoutMgr) noexcept;

    [[nodiscard]] std::expected<void, Error> begin() noexcept;

    using TSampledImageMgrCRef = std::reference_wrapper<const SampledImageManager>;
    using TStorageImageMgrCRef = std::reference_wrapper<const StorageImageManager>;
    void recordSrcPrepareTranfer(std::span<const TSampledImageMgrCRef> srcImageMgrRefs) noexcept;
    void recordSrcPrepareShaderRead(std::span<const TSampledImageMgrCRef> srcImageMgrRefs) noexcept;
    void recordDstPrepareShaderWrite(std::span<const TStorageImageMgrCRef> dstImageMgrRefs) noexcept;
    void recordDispatch(Extent extent, BlockSize blockSize) noexcept;
    void recordDstPrepareTransfer(std::span<const TStorageImageMgrCRef> dstImageMgrRefs) noexcept;

    void recordCopyStagingToSrc(const SampledImageManager& srcImageMgr) noexcept;
    void recordCopyDstToStaging(StorageImageManager& dstImageMgr) noexcept;
    void recordImageCopy(const StorageImageManager& srcImageMgr, SampledImageManager& dstImageMgr) noexcept;

    void recordWaitDownloadComplete(std::span<const TStorageImageMgrCRef> dstImageMgrRefs) noexcept;

    template <typename TQueryPoolManager>
        requires CQueryPoolManager<TQueryPoolManager>
    void recordResetQueryPool(TQueryPoolManager& queryPoolMgr) noexcept;

    [[nodiscard]] std::expected<void, Error> recordTimestampStart(TimestampQueryPoolManager& queryPoolMgr,
                                                                  vk::PipelineStageFlagBits pipelineStage) noexcept;
    [[nodiscard]] std::expected<void, Error> recordTimestampEnd(TimestampQueryPoolManager& queryPoolMgr,
                                                                vk::PipelineStageFlagBits pipelineStage) noexcept;

    [[nodiscard]] std::expected<void, Error> end() noexcept;
    [[nodiscard]] std::expected<void, Error> submitTo(QueueManager& queueMgr, FenceManager& fenceMgr) noexcept;

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;
    std::shared_ptr<CommandPoolManager> pCommandPoolMgr_;

    vk::CommandBuffer commandBuffer_;

    static constexpr vk::ImageSubresourceRange SUBRESOURCE_RANGE{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
};

template <typename TPc>
void CommandBufferManager::pushConstant(const PushConstantManager<TPc>& pushConstantMgr,
                                        const PipelineLayoutManager& pipelineLayoutMgr) noexcept {
    const auto& piplelineLayout = pipelineLayoutMgr.getPipelineLayout();
    commandBuffer_.pushConstants(piplelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(TPc),
                                 pushConstantMgr.getPPushConstant());
}

template <typename TQueryPoolManager>
    requires CQueryPoolManager<TQueryPoolManager>
void CommandBufferManager::recordResetQueryPool(TQueryPoolManager& queryPoolMgr) noexcept {
    auto& queryPool = queryPoolMgr.getQueryPool();
    queryPoolMgr.resetQueryIndex();
    commandBuffer_.resetQueryPool(queryPool, 0, queryPoolMgr.getQueryCount());
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/buffer.cpp"
#endif
