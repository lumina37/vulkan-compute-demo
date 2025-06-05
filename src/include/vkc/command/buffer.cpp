#include <expected>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <utility>

#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/pipeline_layout.hpp"
#include "vkc/query_pool.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/buffer.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

CommandBufferBox::CommandBufferBox(std::shared_ptr<DeviceBox>&& pDeviceBox,
                                   std::shared_ptr<CommandPoolBox>&& pCommandPoolBox,
                                   const vk::CommandBuffer commandBuffer) noexcept
    : pDeviceBox_(std::move(pDeviceBox)), pCommandPoolBox_(std::move(pCommandPoolBox)), commandBuffer_(commandBuffer) {}

CommandBufferBox::CommandBufferBox(CommandBufferBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      pCommandPoolBox_(std::move(rhs.pCommandPoolBox_)),
      commandBuffer_(std::exchange(rhs.commandBuffer_, nullptr)) {}

CommandBufferBox::~CommandBufferBox() noexcept {
    if (commandBuffer_ == nullptr) return;
    vk::Device device = pDeviceBox_->getDevice();
    vk::CommandPool commandPool = pCommandPoolBox_->getCommandPool();
    device.freeCommandBuffers(commandPool, commandBuffer_);
    commandBuffer_ = nullptr;
}

std::expected<CommandBufferBox, Error> CommandBufferBox::create(
    std::shared_ptr<DeviceBox> pDeviceBox, std::shared_ptr<CommandPoolBox> pCommandPoolBox) noexcept {
    vk::Device device = pDeviceBox->getDevice();
    vk::CommandPool commandPool = pCommandPoolBox->getCommandPool();

    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);

    const auto [commandBuffersRes, commandBuffers] = device.allocateCommandBuffers(allocInfo);
    if (commandBuffersRes != vk::Result::eSuccess) {
        return std::unexpected{Error{commandBuffersRes}};
    }
    vk::CommandBuffer commandBuffer = commandBuffers[0];

    return CommandBufferBox{std::move(pDeviceBox), std::move(pCommandPoolBox), commandBuffer};
}

void CommandBufferBox::bindPipeline(PipelineBox& pipelineBox) noexcept {
    commandBuffer_.bindPipeline(pipelineBox.getBindPoint(), pipelineBox.getPipeline());
}

void CommandBufferBox::bindDescSets(DescSetsBox& descSetsBox, const PipelineLayoutBox& pipelineLayoutBox,
                                    const vk::PipelineBindPoint bindPoint) noexcept {
    auto& descSets = descSetsBox.getDescSets();
    commandBuffer_.bindDescriptorSets(bindPoint, pipelineLayoutBox.getPipelineLayout(), 0, (uint32_t)descSets.size(),
                                      descSets.data(), 0, nullptr);
}

std::expected<void, Error> CommandBufferBox::begin() noexcept {
    const auto resetRes = commandBuffer_.reset();
    if (resetRes != vk::Result::eSuccess) {
        return std::unexpected{Error{resetRes}};
    }

    vk::CommandBufferBeginInfo cmdBufBeginInfo;
    cmdBufBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    const auto beginRes = commandBuffer_.begin(cmdBufBeginInfo);
    if (beginRes != vk::Result::eSuccess) {
        return std::unexpected{Error{beginRes}};
    }

    return {};
}

void CommandBufferBox::recordDstPrepareShaderWrite(
    const std::span<const TStorageImageBoxRef> dstImageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eShaderWrite;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eGeneral;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const TStorageImageBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setOldLayout(box.getImageLayout());
        barrier.setImage(box.getImage());

        box.setImageAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = dstImageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

void CommandBufferBox::recordDispatch(const vk::Extent2D extent, const BlockSize blockSize) noexcept {
    const uint32_t groupSizeX = (extent.width + (blockSize.x - 1)) / blockSize.x;
    const uint32_t groupSizeY = (extent.height + (blockSize.y - 1)) / blockSize.y;
    commandBuffer_.dispatch(groupSizeX, groupSizeY, 1);
}

void CommandBufferBox::recordPrepareSendBeforeDispatch(
    const std::span<const TStorageImageBoxRef> dstImageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferRead;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eTransferSrcOptimal;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const TStorageImageBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setOldLayout(box.getImageLayout());
        barrier.setImage(box.getImage());

        box.setImageAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = dstImageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

void CommandBufferBox::recordPrepareSendAfterDispatch(
    const std::span<const TStorageImageBoxRef> dstImageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferRead;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eTransferSrcOptimal;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const TStorageImageBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(box.getImageAccessMask());
        barrier.setOldLayout(box.getImageLayout());
        barrier.setImage(box.getImage());

        box.setImageAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = dstImageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

void CommandBufferBox::recordPreparePresent(std::span<const TPresentImageBoxRef> imageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eMemoryRead;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(SUBRESOURCE_RANGE);

    const auto fillout = [&](const TPresentImageBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(box.getImageAccessMask());
        barrier.setOldLayout(box.getImageLayout());
        barrier.setImage(box.getImage());

        box.setImageAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = imageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

void CommandBufferBox::recordCopyDstToStaging(StorageImageBox& dstImageBox) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageExtent(dstImageBox.getExtent().extent3D());

    commandBuffer_.copyImageToBuffer(dstImageBox.getImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     dstImageBox.getStagingBuffer(), 1, &copyRegion);
}

void CommandBufferBox::recordCopyDstToStagingWithRoi(StorageImageBox& dstImageBox, const Roi roi) noexcept {
    vk::ImageSubresourceLayers subresourceLayers;
    subresourceLayers.setAspectMask(vk::ImageAspectFlagBits::eColor);
    subresourceLayers.setLayerCount(1);
    vk::BufferImageCopy copyRegion;
    const Extent& imageExtent = dstImageBox.getExtent();
    copyRegion.setBufferOffset(imageExtent.calculateBufferOffset(roi.offset()));
    copyRegion.setBufferRowLength(imageExtent.width());
    copyRegion.setBufferImageHeight(imageExtent.height());
    copyRegion.setImageSubresource(subresourceLayers);
    copyRegion.setImageOffset(roi.offset3D());
    copyRegion.setImageExtent(roi.extent3D());

    commandBuffer_.copyImageToBuffer(dstImageBox.getImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     dstImageBox.getStagingBuffer(), 1, &copyRegion);
}

void CommandBufferBox::recordWaitDownloadComplete(const std::span<const TStorageImageBoxRef> dstImageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eHostRead;

    vk::BufferMemoryBarrier barrierTemplate;
    barrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);

    const auto fillout = [&](const TStorageImageBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::BufferMemoryBarrier barrier = barrierTemplate;
        barrier.setBuffer(box.getStagingBuffer());
        barrier.setSize(box.getExtent().size());

        box.setStagingAccessMask(newAccessMask);

        return barrier;
    };

    const auto barriers = dstImageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                                   (vk::DependencyFlags)0, 0, nullptr, (uint32_t)barriers.size(), barriers.data(), 0,
                                   nullptr);
}

std::expected<void, Error> CommandBufferBox::recordTimestampStart(
    TimestampQueryPoolBox& queryPoolBox, const vk::PipelineStageFlagBits pipelineStage) noexcept {
    vk::QueryPool queryPool = queryPoolBox.getQueryPool();
    const int queryIndex = queryPoolBox.getQueryIndex();

    auto addIndexRes = queryPoolBox.addQueryIndex();
    if (!addIndexRes) return std::unexpected{std::move(addIndexRes.error())};

    commandBuffer_.writeTimestamp(pipelineStage, queryPool, queryIndex);

    return {};
}

std::expected<void, Error> CommandBufferBox::recordTimestampEnd(
    TimestampQueryPoolBox& queryPoolBox, const vk::PipelineStageFlagBits pipelineStage) noexcept {
    vk::QueryPool queryPool = queryPoolBox.getQueryPool();
    const int queryIndex = queryPoolBox.getQueryIndex();

    auto addIndexRes = queryPoolBox.addQueryIndex();
    if (!addIndexRes) return std::unexpected{std::move(addIndexRes.error())};

    commandBuffer_.writeTimestamp(pipelineStage, queryPool, queryIndex);

    return {};
}

std::expected<void, Error> CommandBufferBox::end() noexcept {
    const auto endRes = commandBuffer_.end();
    if (endRes != vk::Result::eSuccess) {
        return std::unexpected{Error{endRes}};
    }
    return {};
}

template void CommandBufferBox::recordPrepareReceiveBeforeDispatch<SampledImageBox>(
    std::span<const std::reference_wrapper<SampledImageBox>>) noexcept;
template void CommandBufferBox::recordPrepareReceiveBeforeDispatch<StorageImageBox>(
    std::span<const std::reference_wrapper<StorageImageBox>>) noexcept;

template void CommandBufferBox::recordSrcPrepareShaderRead<SampledImageBox>(
    std::span<const std::reference_wrapper<SampledImageBox>>) noexcept;
template void CommandBufferBox::recordSrcPrepareShaderRead<StorageImageBox>(
    std::span<const std::reference_wrapper<StorageImageBox>>) noexcept;

template void CommandBufferBox::recordCopyStagingToSrc<SampledImageBox>(const SampledImageBox& srcImageBox) noexcept;
template void CommandBufferBox::recordCopyStagingToSrc<StorageImageBox>(const StorageImageBox& srcImageBox) noexcept;

}  // namespace vkc
