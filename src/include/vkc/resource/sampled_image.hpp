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

class SampledImageBox {
    SampledImageBox(ImageBox&& imageBox, ImageViewBox&& imageViewBox, MemoryBox&& imageMemoryBox,
                    const vk::DescriptorImageInfo& descImageInfo) noexcept;

public:
    SampledImageBox(const SampledImageBox&) = delete;
    SampledImageBox(SampledImageBox&& rhs) noexcept = default;
    ~SampledImageBox() noexcept = default;

    [[nodiscard]] static std::expected<SampledImageBox, Error> create(std::shared_ptr<DeviceBox>& pDeviceBox,
                                                                      const Extent& extent) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self.imageBox_).getExtent();
    }

    [[nodiscard]] vk::Image getVkImage() const noexcept { return imageBox_.getVkImage(); }
    [[nodiscard]] vk::AccessFlags getAccessMask() const noexcept { return accessMask_; }
    [[nodiscard]] vk::ImageLayout getImageLayout() const noexcept { return imageLayout_; }
    [[nodiscard]] static constexpr vk::DescriptorType getDescType() noexcept {
        return vk::DescriptorType::eSampledImage;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;
    [[nodiscard]] static constexpr vk::DescriptorSetLayoutBinding draftDescSetLayoutBinding() noexcept;

    [[nodiscard]] std::expected<void, Error> upload(const std::byte* pSrc) noexcept;
    [[nodiscard]] std::expected<void, Error> uploadWithRoi(const std::byte* pSrc, const Roi& roi, size_t bufferOffset,
                                                           size_t bufferRowPitch) noexcept;
    void setAccessMask(vk::AccessFlags accessMask) noexcept { accessMask_ = accessMask; }
    void setImageLayout(vk::ImageLayout imageLayout) noexcept { imageLayout_ = imageLayout; }

private:
    ImageBox imageBox_;
    ImageViewBox imageViewBox_;
    MemoryBox imageMemoryBox_;

    vk::DescriptorImageInfo descImageInfo_;
    vk::AccessFlags accessMask_;
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
