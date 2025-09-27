#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/memory.hpp"

namespace vkc {

class SampledImageBox {
    SampledImageBox(std::shared_ptr<DeviceBox>&& pDeviceBox, Extent extent, vk::Image image, vk::ImageView imageView,
                    MemoryBox&& imageMemoryBox, vk::DescriptorImageInfo descImageInfo) noexcept;

public:
    SampledImageBox(const SampledImageBox&) = delete;
    SampledImageBox(SampledImageBox&& rhs) noexcept;
    ~SampledImageBox() noexcept;

    [[nodiscard]] static std::expected<SampledImageBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                      const Extent& extent) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    [[nodiscard]] vk::Image getVkImage() const noexcept { return image_; }
    [[nodiscard]] vk::AccessFlags getImageAccessMask() const noexcept { return imageAccessMask_; }
    [[nodiscard]] vk::ImageLayout getImageLayout() const noexcept { return imageLayout_; }
    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eSampledImage;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

    [[nodiscard]] std::expected<void, Error> upload(const std::byte* pSrc) noexcept;
    [[nodiscard]] std::expected<void, Error> uploadWithRoi(const std::byte* pSrc, Roi roi,
                                                           size_t bufferRowPitch) noexcept;
    void setImageAccessMask(vk::AccessFlags accessMask) noexcept { imageAccessMask_ = accessMask; }
    void setImageLayout(vk::ImageLayout imageLayout) noexcept { imageLayout_ = imageLayout; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    Extent extent_;

    vk::Image image_;
    vk::ImageView imageView_;
    MemoryBox imageMemoryBox_;

    vk::DescriptorImageInfo descImageInfo_;
    vk::AccessFlags imageAccessMask_;
    vk::ImageLayout imageLayout_;
};

constexpr vk::DescriptorSetLayoutBinding SampledImageBox::draftDescSetLayoutBinding() noexcept {
    vk::DescriptorSetLayoutBinding binding;
    binding.setDescriptorCount(1);
    binding.setDescriptorType(getDescType());
    binding.setStageFlags(vk::ShaderStageFlagBits::eCompute);
    return binding;
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/sampled_image.cpp"
#endif
