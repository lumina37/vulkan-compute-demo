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

class PresentImageBox {
    PresentImageBox(std::shared_ptr<DeviceBox>&& pDeviceBox, const Extent& extent, vk::Image image, vk::ImageView imageView,
                    vk::Buffer stagingBuffer, MemoryBox&& stagingMemoryBox,
                    const vk::DescriptorImageInfo& descImageInfo) noexcept;

public:
    PresentImageBox(const PresentImageBox&) = delete;
    PresentImageBox(PresentImageBox&& rhs) noexcept;
    ~PresentImageBox() noexcept;

    [[nodiscard]] static std::expected<PresentImageBox, Error> create(std::shared_ptr<DeviceBox> pDeviceBox,
                                                                      vk::Image image, const Extent& extent) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self).extent_;
    }

    [[nodiscard]] vk::Image getVkImage() const noexcept { return image_; }
    [[nodiscard]] vk::Buffer getStagingBuffer() const noexcept { return stagingBuffer_; }
    [[nodiscard]] vk::AccessFlags getImageAccessMask() const noexcept { return imageAccessMask_; }
    [[nodiscard]] vk::ImageLayout getImageLayout() const noexcept { return imageLayout_; }
    [[nodiscard]] vk::AccessFlags getStagingAccessMask() const noexcept { return stagingAccessMask_; }

    [[nodiscard]] std::expected<void, Error> upload(const std::byte* pSrc) noexcept;
    [[nodiscard]] std::expected<void, Error> uploadWithRoi(const std::byte* pSrc, Roi roi,
                                                           size_t bufferRowPitch) noexcept;
    void setImageAccessMask(vk::AccessFlags accessMask) noexcept { imageAccessMask_ = accessMask; }
    void setImageLayout(vk::ImageLayout imageLayout) noexcept { imageLayout_ = imageLayout; }
    void setStagingAccessMask(vk::AccessFlags accessMask) noexcept { stagingAccessMask_ = accessMask; }

private:
    std::shared_ptr<DeviceBox> pDeviceBox_;

    Extent extent_;

    vk::Image image_;
    vk::ImageView imageView_;

    vk::Buffer stagingBuffer_;
    MemoryBox stagingMemoryBox_;

    vk::DescriptorImageInfo descImageInfo_;
    vk::AccessFlags imageAccessMask_;
    vk::ImageLayout imageLayout_;
    vk::AccessFlags stagingAccessMask_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/present_image.cpp"
#endif
