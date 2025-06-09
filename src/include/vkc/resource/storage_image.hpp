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

typedef enum StorageImageType {
    Write = 1 << 0,
    Read = 1 << 1,
    ReadWrite = Read | Write,
} StorageImageType;

class StorageImageBox {
    StorageImageBox(std::shared_ptr<DeviceBox>&& pDeviceBox, Extent extent, vk::Image image,
                        vk::ImageView imageView, vk::DeviceMemory imageMemory, vk::Buffer stagingBuffer,
                        vk::DeviceMemory stagingMemory, vk::DescriptorImageInfo descImageInfo) noexcept;

public:
    StorageImageBox(const StorageImageBox&) = delete;
    StorageImageBox(StorageImageBox&& rhs) noexcept;
    ~StorageImageBox() noexcept;

    [[nodiscard]] static std::expected<StorageImageBox, Error> create(
        const PhyDeviceBox& phyDeviceBox, std::shared_ptr<DeviceBox> pDeviceBox, const Extent& extent,
        StorageImageType imageType = StorageImageType::Write) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    [[nodiscard]] vk::Image getImage() const noexcept { return image_; }
    [[nodiscard]] vk::Buffer getStagingBuffer() const noexcept { return stagingBuffer_; }
    [[nodiscard]] vk::AccessFlags getImageAccessMask() const noexcept { return imageAccessMask_; }
    [[nodiscard]] vk::ImageLayout getImageLayout() const noexcept { return imageLayout_; }
    [[nodiscard]] vk::AccessFlags getStagingAccessMask() const noexcept { return stagingAccessMask_; }
    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eStorageImage;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

    [[nodiscard]] std::expected<void, Error> upload(const std::byte* pSrc) noexcept;
    [[nodiscard]] std::expected<void, Error> uploadWithRoi(const std::byte* pSrc, Roi roi,
                                                           size_t bufferRowPitch) noexcept;
    [[nodiscard]] std::expected<void, Error> download(std::byte* pDst) noexcept;
    [[nodiscard]] std::expected<void, Error> downloadWithRoi(std::byte* pDst, Roi roi, size_t bufferRowPitch) noexcept;
    void setImageAccessMask(vk::AccessFlags accessMask) noexcept { imageAccessMask_ = accessMask; }
    void setImageLayout(vk::ImageLayout imageLayout) noexcept { imageLayout_ = imageLayout; }
    void setStagingAccessMask(vk::AccessFlags accessMask) noexcept { stagingAccessMask_ = accessMask; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    Extent extent_;

    vk::Image image_;
    vk::ImageView imageView_;
    vk::DeviceMemory imageMemory_;

    vk::Buffer stagingBuffer_;
    vk::DeviceMemory stagingMemory_;

    vk::DescriptorImageInfo descImageInfo_;
    vk::AccessFlags imageAccessMask_;
    vk::ImageLayout imageLayout_;
    vk::AccessFlags stagingAccessMask_;
};

constexpr vk::DescriptorSetLayoutBinding StorageImageBox::draftDescSetLayoutBinding() noexcept {
    vk::DescriptorSetLayoutBinding binding;
    binding.setDescriptorCount(1);
    binding.setDescriptorType(vk::DescriptorType::eStorageImage);
    binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return binding;
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/storage_image.cpp"
#endif
