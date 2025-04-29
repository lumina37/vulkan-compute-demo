#pragma once

#include <expected>
#include <memory>
#include <utility>

#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/device/queue.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/query_pool.hpp"
#include "vkc/resource/image.hpp"
#include "vkc/resource/push_constant.hpp"

namespace vkc {

struct BlockSize {
    using Tv = uint32_t;
    Tv x, y, z;
};

class CommandBufferManager {
    CommandBufferManager(std::shared_ptr<DeviceManager>&& pDeviceMgr,
                         std::shared_ptr<CommandPoolManager>&& pCommandPoolMgr, vk::CommandBuffer commandBuffer,
                         vk::Fence completeFence) noexcept;

public:
    CommandBufferManager(CommandBufferManager&& rhs) noexcept;
    ~CommandBufferManager() noexcept;

    [[nodiscard]] static std::expected<CommandBufferManager, Error> create(
        std::shared_ptr<DeviceManager> pDeviceMgr, std::shared_ptr<CommandPoolManager> pCommandPoolMgr) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getCommandBuffer(this Self&& self) noexcept {
        return std::forward_like<Self>(self).commandBuffer_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getCompleteFence(this Self&& self) noexcept {
        return std::forward_like<Self>(self).completeFence_;
    }

    void bindPipeline(PipelineManager& pipelineMgr);
    void bindDescSets(DescSetsManager& descSetsMgr, const PipelineLayoutManager& pipelineLayoutMgr);

    template <typename TPc>
    void pushConstant(const PushConstantManager<TPc>& pushConstantMgr, const PipelineLayoutManager& pipelineLayoutMgr);
    std::expected<void, Error> begin();

    using TImageMgrCRef = std::reference_wrapper<const ImageManager>;
    void recordSrcPrepareTranfer(std::span<const TImageMgrCRef> srcImageMgrRefs);
    void recordUploadToSrc(std::span<const TImageMgrCRef> srcImageMgrRefs);

    struct ImageManagerPair {
        const ImageManager& copyFrom;
        const ImageManager& copyTo;
    };
    void recordImageCopy(std::span<const ImageManagerPair> imageMgrPairs);
    void recordSrcPrepareShaderRead(std::span<const TImageMgrCRef> srcImageMgrRefs);
    void recordDstPrepareShaderWrite(std::span<const TImageMgrCRef> dstImageMgrRefs);
    void recordDispatch(Extent extent, BlockSize blockSize);
    void recordDstPrepareTransfer(std::span<const TImageMgrCRef> dstImageMgrRefs);
    void recordDownloadToDst(std::span<const TImageMgrCRef> dstImageMgrRefs);
    void recordWaitDownloadComplete(std::span<const TImageMgrCRef> dstImageMgrRefs);

    template <typename TQueryPoolManager>
        requires CQueryPoolManager<TQueryPoolManager>
    void recordResetQueryPool(TQueryPoolManager& queryPoolMgr);

    void recordTimestampStart(TimestampQueryPoolManager& queryPoolMgr, vk::PipelineStageFlagBits pipelineStage);
    void recordTimestampEnd(TimestampQueryPoolManager& queryPoolMgr, vk::PipelineStageFlagBits pipelineStage);
    void end();
    void submitTo(QueueManager& queueMgr);
    vk::Result waitFence();

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;
    std::shared_ptr<CommandPoolManager> pCommandPoolMgr_;

    vk::CommandBuffer commandBuffer_;
    vk::Fence completeFence_;

    static constexpr vk::ImageSubresourceRange SUBRESOURCE_RANGE{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
};

template <typename TPc>
void CommandBufferManager::pushConstant(const PushConstantManager<TPc>& pushConstantMgr,
                                        const PipelineLayoutManager& pipelineLayoutMgr) {
    const auto& piplelineLayout = pipelineLayoutMgr.getPipelineLayout();
    commandBuffer_.pushConstants(piplelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(TPc),
                                 pushConstantMgr.getPPushConstant());
}

template <typename TQueryPoolManager>
    requires CQueryPoolManager<TQueryPoolManager>
void CommandBufferManager::recordResetQueryPool(TQueryPoolManager& queryPoolMgr) {
    auto& queryPool = queryPoolMgr.getQueryPool();
    queryPoolMgr.resetQueryIndex();
    commandBuffer_.resetQueryPool(queryPool, 0, queryPoolMgr.getQueryCount());
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/buffer.cpp"
#endif
