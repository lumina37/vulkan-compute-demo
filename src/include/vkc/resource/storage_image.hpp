#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

typedef enum StorageImageType {
    Write = 1 << 0,
    Read = 1 << 1,
} StorageImageType;

class StorageImageManager {
    StorageImageManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, Extent extent, vk::Image image,
                        vk::ImageView imageView, vk::DeviceMemory imageMemory, vk::Buffer stagingBuffer,
                        vk::DeviceMemory stagingMemory, vk::DescriptorImageInfo descImageInfo) noexcept;

public:
    StorageImageManager(StorageImageManager&& rhs) noexcept;
    ~StorageImageManager() noexcept;

    [[nodiscard]] static std::expected<StorageImageManager, Error> create(
        const PhyDeviceManager& phyDeviceMgr, std::shared_ptr<DeviceManager> pDeviceMgr, const Extent& extent,
        StorageImageType imageType = StorageImageType::Write) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    [[nodiscard]] vk::Image getImage() const noexcept { return image_; }
    [[nodiscard]] vk::Buffer getStagingBuffer() const noexcept { return stagingBuffer_; }
    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eStorageImage;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() const noexcept;

    [[nodiscard]] std::expected<void, Error> uploadFrom(const std::byte* pData, Roi roi) noexcept;
    [[nodiscard]] std::expected<void, Error> downloadTo(std::byte* pData, Roi roi) noexcept;

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

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/storage_image.cpp"
#endif
