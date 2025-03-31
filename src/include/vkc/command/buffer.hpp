#pragma once

#include <array>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/device/queue.hpp"
#include "vkc/extent.hpp"
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
public:
    CommandBufferManager(DeviceManager& deviceMgr, CommandPoolManager& commandPoolMgr);
    ~CommandBufferManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getCommandBuffer(this Self&& self) noexcept {
        return std::forward_like<Self>(self).commandBuffer_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getCompleteFence(this Self&& self) noexcept {
        return std::forward_like<Self>(self).completeFence_;
    }

    void bindPipeline(PipelineManager& pipelineMgr);
    void bindDescSet(DescSetManager& descSetMgr, const PipelineLayoutManager& pipelineLayoutMgr);

    template <typename TPc>
    void pushConstant(const PushConstantManager<TPc>& pushConstantMgr, const PipelineLayoutManager& pipelineLayoutMgr);
    void begin();

    using TImageManagerRef = std::reference_wrapper<const ImageManager>;
    void recordSrcPrepareTranfer(std::span<TImageManagerRef> srcImageMgrRefs);
    void recordUploadToSrc(std::span<TImageManagerRef> srcImageMgrRefs);
    using TImageManagerRefPair = std::array<TImageManagerRef, 2>;
    void recordImageCopy(std::span<TImageManagerRefPair> imageMgrRefPairs);
    void recordSrcPrepareShaderRead(std::span<TImageManagerRef> srcImageMgrRefs);
    void recordDstPrepareShaderWrite(std::span<TImageManagerRef> dstImageMgrRefs);
    void recordDispatch(ExtentManager extent, BlockSize blockSize);
    void recordDstPrepareTransfer(std::span<TImageManagerRef> dstImageMgrRefs);
    void recordDownloadToDst(std::span<TImageManagerRef> dstImageMgrRefs);
    void recordWaitDownloadComplete(std::span<TImageManagerRef> dstImageMgrRefs);

    template <typename TQueryPoolManager>
        requires CQueryPoolManager<TQueryPoolManager>
    void recordResetQueryPool(TQueryPoolManager& queryPoolMgr);

    void recordTimestampStart(TimestampQueryPoolManager& queryPoolMgr, vk::PipelineStageFlagBits pipelineStage);
    void recordTimestampEnd(TimestampQueryPoolManager& queryPoolMgr, vk::PipelineStageFlagBits pipelineStage);
    void end();
    void submitTo(QueueManager& queueMgr);
    vk::Result waitFence();

private:
    DeviceManager& deviceMgr_;            // FIXME: UAF
    CommandPoolManager& commandPoolMgr_;  // FIXME: UAF
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
