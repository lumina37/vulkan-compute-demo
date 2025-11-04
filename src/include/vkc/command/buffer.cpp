#include <functional>
#include <memory>
#include <ranges>
#include <span>

#include "vkc/command/pool.hpp"
#include "vkc/descriptor/set.hpp"
#include "vkc/device/logical.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/pipeline.hpp"
#include "vkc/query_pool.hpp"
#include "vkc/resource.hpp"

#ifndef _VKC_LIB_HEADER_ONLY
#    include "vkc/command/buffer.hpp"
#endif

namespace vkc {

namespace rgs = std::ranges;

CommandBufferBox::CommandBufferBox(std::shared_ptr<DeviceBox>&& pDeviceBox,
                                   std::shared_ptr<CommandPoolBox>&& pCommandPoolBox,
                                   const vk::CommandBuffer commandBuffer) noexcept
    : pDeviceBox_(std::move(pDeviceBox)),
      pCommandPoolBox_(std::move(pCommandPoolBox)),
      commandBuffer_(commandBuffer),
      dispatchRecorded_(false) {}

CommandBufferBox::CommandBufferBox(CommandBufferBox&& rhs) noexcept
    : pDeviceBox_(std::move(rhs.pDeviceBox_)),
      pCommandPoolBox_(std::move(rhs.pCommandPoolBox_)),
      commandBuffer_(std::exchange(rhs.commandBuffer_, nullptr)),
      dispatchRecorded_(rhs.dispatchRecorded_) {}

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
        return std::unexpected{Error{ECate::eVk, commandBuffersRes}};
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
        return std::unexpected{Error{ECate::eVk, resetRes}};
    }

    vk::CommandBufferBeginInfo cmdBufBeginInfo;
    cmdBufBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
    const auto beginRes = commandBuffer_.begin(cmdBufBeginInfo);
    if (beginRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, beginRes}};
    }

    return {};
}

void CommandBufferBox::recordPrepareShaderWrite(const std::span<const TStorageImageBoxRef> imageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eShaderWrite;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eGeneral;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(_hp::SUBRESOURCE_RANGE);

    const auto fillout = [&](const TStorageImageBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setOldLayout(box.getImageLayout());
        barrier.setImage(box.getVkImage());

        box.setAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = imageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

void CommandBufferBox::recordPrepareShaderWrite(std::span<const TStorageBufferBoxRef> bufferBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eShaderWrite;

    vk::BufferMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);

    const auto fillout = [&](const TStorageBufferBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::BufferMemoryBarrier barrier = barrierTemplate;
        barrier.setBuffer(box.getVkBuffer());
        barrier.setSize(box.getSize());

        box.setAccessMask(newAccessMask);

        return barrier;
    };

    const auto barriers = bufferBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                                   (vk::DependencyFlags)0, 0, nullptr, (uint32_t)barriers.size(), barriers.data(), 0,
                                   nullptr);
}

void CommandBufferBox::recordDispatch(int groupNumX, int groupNumY) noexcept {
    commandBuffer_.dispatch(groupNumX, groupNumY, 1);
    dispatchRecorded_ = true;
}

void CommandBufferBox::recordPrepareSend(const std::span<const TStorageImageBoxRef> imageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferRead;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::eTransferSrcOptimal;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(_hp::SUBRESOURCE_RANGE);

    const auto fillout = [&](const TStorageImageBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setOldLayout(box.getImageLayout());
        if (dispatchRecorded_) {
            barrier.setSrcAccessMask(box.getAccessMask());
        }
        barrier.setImage(box.getVkImage());

        box.setAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = imageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    vk::PipelineStageFlags srcStageMask;
    if (dispatchRecorded_) {
        srcStageMask = vk::PipelineStageFlagBits::eComputeShader;
    } else {
        srcStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
    }

    commandBuffer_.pipelineBarrier(srcStageMask, vk::PipelineStageFlagBits::eTransfer, (vk::DependencyFlags)0, 0,
                                   nullptr, 0, nullptr, (uint32_t)barriers.size(), barriers.data());
}

void CommandBufferBox::recordPrepareSend(std::span<const TStorageBufferBoxRef> bufferBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eTransferRead;

    vk::BufferMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);

    const auto fillout = [&](const TStorageBufferBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::BufferMemoryBarrier barrier = barrierTemplate;
        if (dispatchRecorded_) {
            barrier.setSrcAccessMask(box.getAccessMask());
        }
        barrier.setBuffer(box.getVkBuffer());
        barrier.setSize(box.getSize());

        box.setAccessMask(newAccessMask);

        return barrier;
    };

    const auto barriers = bufferBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    vk::PipelineStageFlags srcStageMask;
    if (dispatchRecorded_) {
        srcStageMask = vk::PipelineStageFlagBits::eComputeShader;
    } else {
        srcStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
    }

    commandBuffer_.pipelineBarrier(srcStageMask, vk::PipelineStageFlagBits::eTransfer, (vk::DependencyFlags)0, 0,
                                   nullptr, (uint32_t)barriers.size(), barriers.data(), 0, nullptr);
}

void CommandBufferBox::recordPreparePresent(std::span<const TPresentImageBoxRef> imageBoxRefs) noexcept {
    constexpr vk::AccessFlags newAccessMask = vk::AccessFlagBits::eMemoryRead;
    constexpr vk::ImageLayout newImageLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::ImageMemoryBarrier barrierTemplate;
    barrierTemplate.setDstAccessMask(newAccessMask);
    barrierTemplate.setNewLayout(newImageLayout);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setSubresourceRange(_hp::SUBRESOURCE_RANGE);

    const auto fillout = [&](const TPresentImageBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::ImageMemoryBarrier barrier = barrierTemplate;
        barrier.setSrcAccessMask(box.getAccessMask());
        barrier.setOldLayout(box.getImageLayout());
        barrier.setImage(box.getVkImage());

        box.setAccessMask(newAccessMask);
        box.setImageLayout(newImageLayout);

        return barrier;
    };

    const auto barriers = imageBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

    commandBuffer_.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe,
                                   (vk::DependencyFlags)0, 0, nullptr, 0, nullptr, (uint32_t)barriers.size(),
                                   barriers.data());
}

void CommandBufferBox::recordCopyImageToStaging(const StorageImageBox& imageBox,
                                                StagingBufferBox& stagingBufferBox) noexcept {
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource(_hp::SUBRESOURCE_LAYERS);
    copyRegion.setImageExtent(imageBox.getExtent().extent3D());

    commandBuffer_.copyImageToBuffer(imageBox.getVkImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     stagingBufferBox.getVkBuffer(), 1, &copyRegion);
}

void CommandBufferBox::recordCopyImageToStagingWithRoi(const StorageImageBox& imageBox,
                                                       StagingBufferBox& stagingBufferBox, const Roi& roi) noexcept {
    vk::BufferImageCopy copyRegion;
    const Extent& imageExtent = imageBox.getExtent();
    copyRegion.setBufferOffset(imageExtent.calculateBufferOffset(roi.offset()));
    copyRegion.setBufferRowLength(imageExtent.width());
    copyRegion.setBufferImageHeight(imageExtent.height());
    copyRegion.setImageSubresource(_hp::SUBRESOURCE_LAYERS);
    copyRegion.setImageOffset(roi.offset3D());
    copyRegion.setImageExtent(roi.extent3D());

    commandBuffer_.copyImageToBuffer(imageBox.getVkImage(), vk::ImageLayout::eTransferSrcOptimal,
                                     stagingBufferBox.getVkBuffer(), 1, &copyRegion);
}

void CommandBufferBox::recordCopyBufferToStaging(const StorageBufferBox& bufferBox,
                                                 StagingBufferBox& stagingBufferBox) noexcept {
    vk::BufferCopy copyRegion;
    copyRegion.setSize(bufferBox.getSize());

    commandBuffer_.copyBuffer(bufferBox.getVkBuffer(), stagingBufferBox.getVkBuffer(), 1, &copyRegion);
}

void CommandBufferBox::recordWaitDownloadComplete(
    const std::span<const TStagingBufferBoxRef> stagingBufferBoxRefs) noexcept {
    vk::BufferMemoryBarrier barrierTemplate;
    barrierTemplate.setSrcAccessMask(vk::AccessFlagBits::eNone);
    barrierTemplate.setDstAccessMask(vk::AccessFlagBits::eHostRead);
    barrierTemplate.setSrcQueueFamilyIndex(vk::QueueFamilyIgnored);
    barrierTemplate.setDstQueueFamilyIndex(vk::QueueFamilyIgnored);

    const auto fillout = [&](const TStagingBufferBoxRef boxRef) {
        auto& box = boxRef.get();

        vk::BufferMemoryBarrier barrier = barrierTemplate;
        barrier.setBuffer(box.getVkBuffer());
        barrier.setSize(box.getSize());

        return barrier;
    };

    const auto barriers = stagingBufferBoxRefs | rgs::views::transform(fillout) | rgs::to<std::vector>();

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

std::expected<void, Error> CommandBufferBox::recordPerfQueryStart(PerfQueryPoolBox& queryPoolBox) noexcept {
    vk::QueryPool queryPool = queryPoolBox.getQueryPool();
    const int queryIndex = queryPoolBox.getQueryIndex();

    commandBuffer_.beginQuery(queryPool, queryIndex, vk::QueryControlFlags(0));

    return {};
}

std::expected<void, Error> CommandBufferBox::recordPerfQueryEnd(PerfQueryPoolBox& queryPoolBox) noexcept {
    vk::QueryPool queryPool = queryPoolBox.getQueryPool();

    const int queryIndex = queryPoolBox.getQueryIndex();
    commandBuffer_.endQuery(queryPool, queryIndex);

    auto addIndexRes = queryPoolBox.addQueryIndex();
    if (!addIndexRes) return std::unexpected{std::move(addIndexRes.error())};

    return {};
}

std::expected<void, Error> CommandBufferBox::end() noexcept {
    const auto endRes = commandBuffer_.end();
    if (endRes != vk::Result::eSuccess) {
        return std::unexpected{Error{ECate::eVk, endRes}};
    }
    return {};
}

template void CommandBufferBox::recordPrepareReceive<SampledImageBox>(
    std::span<const std::reference_wrapper<SampledImageBox>>) noexcept;
template void CommandBufferBox::recordPrepareReceive<StorageImageBox>(
    std::span<const std::reference_wrapper<StorageImageBox>>) noexcept;
template void CommandBufferBox::recordPrepareReceive<UniformBufferBox>(
    std::span<const std::reference_wrapper<UniformBufferBox>>) noexcept;
template void CommandBufferBox::recordPrepareReceive<StorageBufferBox>(
    std::span<const std::reference_wrapper<StorageBufferBox>>) noexcept;

template void CommandBufferBox::recordPrepareShaderRead<SampledImageBox>(
    std::span<const std::reference_wrapper<SampledImageBox>>) noexcept;
template void CommandBufferBox::recordPrepareShaderRead<StorageImageBox>(
    std::span<const std::reference_wrapper<StorageImageBox>>) noexcept;
template void CommandBufferBox::recordPrepareShaderRead<UniformBufferBox>(
    std::span<const std::reference_wrapper<UniformBufferBox>>) noexcept;
template void CommandBufferBox::recordPrepareShaderRead<StorageBufferBox>(
    std::span<const std::reference_wrapper<StorageBufferBox>>) noexcept;

template void CommandBufferBox::recordCopyStagingToImage<SampledImageBox>(const StagingBufferBox&,
                                                                          SampledImageBox&) noexcept;
template void CommandBufferBox::recordCopyStagingToImage<StorageImageBox>(const StagingBufferBox&,
                                                                          StorageImageBox&) noexcept;
template void CommandBufferBox::recordCopyStagingToImageWithRoi<SampledImageBox>(const StagingBufferBox&,
                                                                                 SampledImageBox&, const Roi&) noexcept;
template void CommandBufferBox::recordCopyStagingToImageWithRoi<StorageImageBox>(const StagingBufferBox&,
                                                                                 StorageImageBox&, const Roi&) noexcept;

template void CommandBufferBox::recordCopyStagingToBuffer<UniformBufferBox>(const StagingBufferBox&,
                                                                            UniformBufferBox&) noexcept;
template void CommandBufferBox::recordCopyStagingToBuffer<StorageBufferBox>(const StagingBufferBox&,
                                                                            StorageBufferBox&) noexcept;

template void CommandBufferBox::recordCopyStorageToAnother<SampledImageBox>(const StorageImageBox&,
                                                                            SampledImageBox&) noexcept;
template void CommandBufferBox::recordCopyStorageToAnother<StorageImageBox>(const StorageImageBox&,
                                                                            StorageImageBox&) noexcept;
template void CommandBufferBox::recordCopyStorageToAnotherWithRoi<SampledImageBox>(const StorageImageBox&,
                                                                                   SampledImageBox&,
                                                                                   const Roi&) noexcept;
template void CommandBufferBox::recordCopyStorageToAnotherWithRoi<StorageImageBox>(const StorageImageBox&,
                                                                                   StorageImageBox&,
                                                                                   const Roi&) noexcept;
}  // namespace vkc
