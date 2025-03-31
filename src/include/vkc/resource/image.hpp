#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

#include <vulkan/vulkan.hpp>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"

namespace vkc {

enum class ImageType {
    Read,
    Write,
};

class ImageManager {
public:
    ImageManager(const PhysicalDeviceManager& phyDeviceMgr, const std::shared_ptr<DeviceManager>& pDeviceMgr,
                 const ExtentManager& extent, ImageType imageType);
    ~ImageManager() noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getImage(this Self&& self) noexcept {
        return std::forward_like<Self>(self).image_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getImageView(this Self&& self) noexcept {
        return std::forward_like<Self>(self).imageView_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getStagingBuffer(this Self&& self) noexcept {
        return std::forward_like<Self>(self).stagingBuffer_;
    }

    [[nodiscard]] ImageType getImageType() const noexcept { return imageType_; }
    [[nodiscard]] vk::DescriptorType getDescType() const noexcept { return descType_; }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() const noexcept;

    vk::Result uploadFrom(std::span<std::byte> data);
    vk::Result downloadTo(std::span<std::byte> data);

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    ExtentManager extent_;
    ImageType imageType_;
    vk::DescriptorType descType_;

    vk::Image image_;
    vk::ImageView imageView_;
    vk::DeviceMemory imageMemory_;

    vk::Buffer stagingBuffer_;
    vk::DeviceMemory stagingMemory_;

    vk::DescriptorImageInfo descImageInfo_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/image.cpp"
#endif
