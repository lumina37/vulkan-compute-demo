#pragma once

#include <expected>
#include <memory>
#include <utility>

#include "vkc/device.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/image.hpp"
#include "vkc/resource/image_view.hpp"

namespace vkc {

class PresentImageBox {
    PresentImageBox(ImageBox&& imageBox, ImageViewBox&& imageViewBox) noexcept;

public:
    PresentImageBox(const PresentImageBox&) = delete;
    PresentImageBox(PresentImageBox&& rhs) noexcept = default;
    ~PresentImageBox() noexcept = default;

    [[nodiscard]] static std::expected<PresentImageBox, Error> create(std::shared_ptr<DeviceBox>& pDeviceBox,
                                                                      ImageBox& imageBox) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getExtent(this Self&& self) noexcept {
        return std::forward_like<Self>(self.imageBox_).getExtent();
    }

    [[nodiscard]] vk::Image getVkImage() const noexcept { return imageBox_.getVkImage(); }
    [[nodiscard]] vk::AccessFlags getAccessMask() const noexcept { return accessMask_; }
    [[nodiscard]] vk::ImageLayout getImageLayout() const noexcept { return imageLayout_; }

    void setAccessMask(vk::AccessFlags accessMask) noexcept { accessMask_ = accessMask; }
    void setImageLayout(vk::ImageLayout imageLayout) noexcept { imageLayout_ = imageLayout; }

private:
    ImageBox imageBox_;
    ImageViewBox imageViewBox_;

    vk::AccessFlags accessMask_;
    vk::ImageLayout imageLayout_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/present_image.cpp"
#endif
