#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

class SampledImageManager {
    SampledImageManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, Extent extent, vk::Image image,
                        vk::ImageView imageView, vk::DeviceMemory imageMemory, vk::Buffer stagingBuffer,
                        vk::DeviceMemory stagingMemory, vk::DescriptorImageInfo descImageInfo) noexcept;

public:
    SampledImageManager(SampledImageManager&& rhs) noexcept;
    ~SampledImageManager() noexcept;

    [[nodiscard]] static std::expected<SampledImageManager, Error> create(const PhyDeviceManager& phyDeviceMgr,
                                                                          std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                          const Extent& extent) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    [[nodiscard]] vk::Image getImage() const noexcept { return image_; }
    [[nodiscard]] vk::Buffer getStagingBuffer() const noexcept { return stagingBuffer_; }
    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eSampledImage;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

    [[nodiscard]] std::expected<void, Error> upload(const std::byte* pSrc) noexcept;
    [[nodiscard]] std::expected<void, Error> uploadWithRoi(const std::byte* pSrc, Roi roi) noexcept;

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    Extent extent_;

    vk::Image image_;
    vk::ImageView imageView_;
    vk::DeviceMemory imageMemory_;

    vk::Buffer stagingBuffer_;
    vk::DeviceMemory stagingMemory_;

    vk::DescriptorImageInfo descImageInfo_;
};

constexpr vk::DescriptorSetLayoutBinding SampledImageManager::draftDescSetLayoutBinding() noexcept {
    vk::DescriptorSetLayoutBinding binding;
    binding.setDescriptorCount(1);
    binding.setDescriptorType(vk::DescriptorType::eSampledImage);
    binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return binding;
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/sampled_image.cpp"
#endif
