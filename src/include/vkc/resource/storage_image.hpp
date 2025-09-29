#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/image.hpp"
#include "vkc/resource/image_view.hpp"
#include "vkc/resource/memory.hpp"

namespace vkc {

class StorageImageBox {
    StorageImageBox(ImageBox&& imageBox, ImageViewBox&& imageViewBox, MemoryBox&& imageMemoryBox,
                    const vk::DescriptorImageInfo& descImageInfo) noexcept;

public:
    StorageImageBox(const StorageImageBox&) = delete;
    StorageImageBox(StorageImageBox&& rhs) noexcept = default;
    ~StorageImageBox() noexcept = default;

    [[nodiscard]] static std::expected<StorageImageBox, Error> create(
        std::shared_ptr<DeviceBox>& pDeviceBox, const Extent& extent,
        StorageType imageType = StorageType::ReadWrite) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self.imageBox_).getExtent();
    }

    [[nodiscard]] vk::Image getVkImage() const noexcept { return imageBox_.getVkImage(); }
    [[nodiscard]] vk::AccessFlags getImageAccessMask() const noexcept { return imageAccessMask_; }
    [[nodiscard]] vk::ImageLayout getImageLayout() const noexcept { return imageLayout_; }
    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eStorageImage;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

    [[nodiscard]] std::expected<void, Error> upload(const std::byte* pSrc) noexcept;
    [[nodiscard]] std::expected<void, Error> uploadWithRoi(const std::byte* pSrc, const Roi& roi, size_t bufferOffset,
                                                           size_t bufferRowPitch) noexcept;
    [[nodiscard]] std::expected<void, Error> download(std::byte* pDst) noexcept;
    [[nodiscard]] std::expected<void, Error> downloadWithRoi(std::byte* pDst, const Roi& roi, size_t bufferOffset,
                                                             size_t bufferRowPitch) noexcept;
    void setImageAccessMask(vk::AccessFlags accessMask) noexcept { imageAccessMask_ = accessMask; }
    void setImageLayout(vk::ImageLayout imageLayout) noexcept { imageLayout_ = imageLayout; }

private:
    ImageBox imageBox_;
    ImageViewBox imageViewBox_;
    MemoryBox imageMemoryBox_;

    vk::DescriptorImageInfo descImageInfo_;
    vk::AccessFlags imageAccessMask_;
    vk::ImageLayout imageLayout_;
};

constexpr vk::DescriptorSetLayoutBinding StorageImageBox::draftDescSetLayoutBinding() noexcept {
    vk::DescriptorSetLayoutBinding binding;
    binding.setDescriptorCount(1);
    binding.setDescriptorType(getDescType());
    binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return binding;
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/storage_image.cpp"
#endif
