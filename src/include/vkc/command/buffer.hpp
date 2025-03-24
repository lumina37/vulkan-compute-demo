#pragma once

#include <array>
#include <type_traits>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/extent.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/query_pool.hpp"
#include "vkc/queue.hpp"
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

    void begin();
    void bindPipeline(PipelineManager& pipelineMgr);
    void bindDescSet(DescSetManager& descSetMgr, const PipelineLayoutManager& pipelineLayoutMgr);

    template <typename TPc>
    void pushConstant(const PushConstantManager<TPc>& pushConstantMgr, const PipelineLayoutManager& pipelineLayoutMgr);

    template <typename... TMgr>
        requires(std::is_same_v<TMgr, ImageManager> && ...)
    void recordSrcPrepareTranfer(TMgr&... srcImageMgrs);

    template <typename... TMgr>
        requires(std::is_same_v<TMgr, ImageManager> && ...)
    void recordUploadToSrc(TMgr&... srcImageMgrs);

    template <typename... TMgrPair>
        requires(std::is_same_v<TMgrPair, std::array<std::reference_wrapper<ImageManager>, 2>> && ...)
    void recordImageCopy(TMgrPair... imageMgrPairs);

    template <typename... TMgr>
        requires(std::is_same_v<TMgr, ImageManager> && ...)
    void recordSrcPrepareShaderRead(TMgr&... srcImageMgrs);

    template <typename... TMgr>
        requires(std::is_same_v<TMgr, ImageManager> && ...)
    void recordDstPrepareShaderWrite(TMgr&... dstImageMgrs);

    void recordDispatch(ExtentManager extent, BlockSize blockSize);

    template <typename... TMgr>
        requires(std::is_same_v<TMgr, ImageManager> && ...)
    void recordDstPrepareTransfer(TMgr&... dstImageMgrs);

    template <typename... TMgr>
        requires(std::is_same_v<TMgr, ImageManager> && ...)
    void recordDownloadToDst(TMgr&... dstImageMgrs);

    template <typename... TMgr>
        requires(std::is_same_v<TMgr, ImageManager> && ...)
    void recordWaitDownloadComplete(TMgr&... dstImageMgrs);

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

template <typename... TMgr>
    requires(std::is_same_v<TMgr, ImageManager> && ...)
void CommandBufferManager::recordSrcPrepareTranfer(TMgr&... srcImageMgrs) {
    vk::ImageMemoryBarrier uploadConvBarrierTemplate;
    uploadConvBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    uploadConvBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    uploadConvBarrierTemplate.setOldLayout(vk::ImageLayout::eUndefined);
    uploadConvBarrierTemplate.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    uploadConvBarrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto genUploadConvBarrier = [uploadConvBarrierTemplate](auto& mgr) {
        vk::ImageMemoryBarrier uploadConvBarrier = uploadConvBarrierTemplate;
        uploadConvBarrier.setImage(mgr.getImage());
        return uploadConvBarrier;
    };
    std::array uploadConvBarriers{genUploadConvBarrier(srcImageMgrs)...};

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, uploadConvBarriers.size(),
                                   uploadConvBarriers.data());
}

template <typename... TMgr>
    requires(std::is_same_v<TMgr, ImageManager> && ...)
void CommandBufferManager::recordUploadToSrc(TMgr&... srcImageMgrs) {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegionTemplate;
    copyRegionTemplate.setImageSubresource(subresourceLayers);

    const auto copyBufferToImage = [&](auto& mgr) {
        vk::BufferImageCopy copyRegion = copyRegionTemplate;
        copyRegion.setImageExtent(mgr.getExtent().extent3D());
        commandBuffer_.copyBufferToImage(mgr.getStagingBuffer(), mgr.getImage(), vk::ImageLayout::eTransferDstOptimal,
                                         1, &copyRegion);
    };
    (copyBufferToImage(srcImageMgrs), ...);
}

template <typename... TMgrPair>
    requires(std::is_same_v<TMgrPair, std::array<std::reference_wrapper<ImageManager>, 2>> && ...)
void CommandBufferManager::recordImageCopy(TMgrPair... imageMgrPairs) {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::ImageCopy copyRegionTemplate;
    copyRegionTemplate.setSrcSubresource(subresourceLayers);
    copyRegionTemplate.setDstSubresource(subresourceLayers);

    const auto copyBufferToImage = [&](auto& mgrPair) {
        vk::ImageCopy copyRegion = copyRegionTemplate;
        copyRegion.setExtent(mgrPair[0].get().getExtent().extent3D());
        commandBuffer_.copyImage(mgrPair[0].get().getImage(), vk::ImageLayout::eTransferSrcOptimal,
                                 mgrPair[1].get().getImage(), vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);
    };
    (copyBufferToImage(imageMgrPairs), ...);
}

template <typename... TMgr>
    requires(std::is_same_v<TMgr, ImageManager> && ...)
void CommandBufferManager::recordSrcPrepareShaderRead(TMgr&... srcImageMgrs) {
    vk::ImageMemoryBarrier shaderCompatibleBarrierTemplate;
    shaderCompatibleBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    shaderCompatibleBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
    shaderCompatibleBarrierTemplate.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    shaderCompatibleBarrierTemplate.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    shaderCompatibleBarrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto genShaderCompatibleBarrier = [shaderCompatibleBarrierTemplate](auto& mgr) {
        vk::ImageMemoryBarrier shaderCompatibleBarrier = shaderCompatibleBarrierTemplate;
        shaderCompatibleBarrier.setImage(mgr.getImage());
        return shaderCompatibleBarrier;
    };
    std::array shaderCompatibleBarriers{genShaderCompatibleBarrier(srcImageMgrs)...};

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, shaderCompatibleBarriers.size(),
                                   shaderCompatibleBarriers.data());
}

template <typename... TMgr>
    requires(std::is_same_v<TMgr, ImageManager> && ...)
void CommandBufferManager::recordDstPrepareShaderWrite(TMgr&... dstImageMgrs) {
    vk::ImageSubresourceRange subresourceRange;
    subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceRange.setLevelCount(1);
    subresourceRange.setLayerCount(1);

    // Shader Compatible Image Layout
    vk::ImageMemoryBarrier shaderCompatibleBarrierTemplate;
    shaderCompatibleBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    shaderCompatibleBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eShaderWrite);
    shaderCompatibleBarrierTemplate.setOldLayout(vk::ImageLayout::eUndefined);
    shaderCompatibleBarrierTemplate.setNewLayout(vk::ImageLayout::eGeneral);
    shaderCompatibleBarrierTemplate.setSubresourceRange(subresourceRange);

    const auto genShaderCompatibleBarrier = [shaderCompatibleBarrierTemplate](auto& mgr) {
        vk::ImageMemoryBarrier shaderCompatibleBarrier = shaderCompatibleBarrierTemplate;
        shaderCompatibleBarrier.setImage(mgr.getImage());
        return shaderCompatibleBarrier;
    };
    std::array shaderCompatibleBarriers{genShaderCompatibleBarrier(dstImageMgrs)...};

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, shaderCompatibleBarriers.size(),
                                   shaderCompatibleBarriers.data());
}

template <typename... TMgr>
    requires(std::is_same_v<TMgr, ImageManager> && ...)
void CommandBufferManager::recordDstPrepareTransfer(TMgr&... dstImageMgrs) {
    vk::ImageMemoryBarrier downloadConvBarrierTemplate;
    downloadConvBarrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eShaderWrite);
    downloadConvBarrierTemplate.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    downloadConvBarrierTemplate.setOldLayout(vk::ImageLayout::eGeneral);
    downloadConvBarrierTemplate.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    downloadConvBarrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto genDownloadConvBarrier = [downloadConvBarrierTemplate](auto& mgr) {
        vk::ImageMemoryBarrier downloadConvBarrier = downloadConvBarrierTemplate;
        downloadConvBarrier.setImage(mgr.getImage());
        return downloadConvBarrier;
    };
    std::array downloadConvBarriers{genDownloadConvBarrier(dstImageMgrs)...};

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, downloadConvBarriers.size(),
                                   downloadConvBarriers.data());
}

template <typename... TMgr>
    requires(std::is_same_v<TMgr, ImageManager> && ...)
void CommandBufferManager::recordDownloadToDst(TMgr&... dstImageMgrs) {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegionTemplate;
    copyRegionTemplate.setImageSubresource(subresourceLayers);

    const auto copyImageToBuffer = [&](auto& mgr) {
        vk::BufferImageCopy copyRegion = copyRegionTemplate;
        copyRegion.setImageExtent(mgr.getExtent().extent3D());
        commandBuffer_.copyImageToBuffer(mgr.getImage(), vk::ImageLayout::eTransferSrcOptimal, mgr.getStagingBuffer(),
                                         1, &copyRegion);
    };
    (copyImageToBuffer(dstImageMgrs), ...);
}

template <typename... TMgr>
    requires(std::is_same_v<TMgr, ImageManager> && ...)
void CommandBufferManager::recordWaitDownloadComplete(TMgr&... dstImageMgrs) {
    const auto genDownloadCompleteBarrier = [](auto& mgr) {
        vk::BufferMemoryBarrier downloadCompleteBarrier;
        downloadCompleteBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
        downloadCompleteBarrier.setDstAccessMask(vk::AccessFlagBits::eHostRead);
        downloadCompleteBarrier.setBuffer(mgr.getStagingBuffer());
        downloadCompleteBarrier.setSize(mgr.getExtent().size());
        return downloadCompleteBarrier;
    };
    std::array downloadCompleteBarriers{genDownloadCompleteBarrier(dstImageMgrs)...};

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                                   (vk::DependencyFlags)0, 0, nullptr, downloadCompleteBarriers.size(),
                                   downloadCompleteBarriers.data(), 0, nullptr);
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
