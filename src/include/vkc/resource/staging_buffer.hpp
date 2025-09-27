#pragma once

#include <cstddef>
#include <expected>
#include <memory>

#include "vkc/device.hpp"
#include "vkc/extent.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"
#include "vkc/resource/buffer.hpp"
#include "vkc/resource/memory.hpp"

namespace vkc {

class StagingBufferBox {
    StagingBufferBox(BufferBox&& bufferBox, MemoryBox&& memoryBox) noexcept;

public:
    StagingBufferBox(const StagingBufferBox&) = delete;
    StagingBufferBox(StagingBufferBox&& rhs) noexcept = default;
    ~StagingBufferBox() noexcept = default;

    [[nodiscard]] static std::expected<StagingBufferBox, Error> create(
        std::shared_ptr<DeviceBox>& pDeviceBox, vk::DeviceSize size,
        StorageType bufferType = StorageType::ReadWrite) noexcept;

    template <typename Self>
    [[nodiscard]] auto&& getBufferBox(this Self&& self) noexcept {
        return std::forward_like<Self>(self).bufferBox_;
    }

    [[nodiscard]] vk::DeviceSize getSize() const noexcept { return bufferBox_.getSize(); }

    [[nodiscard]] std::expected<void, Error> upload(const std::byte* pSrc) noexcept;
    [[nodiscard]] std::expected<void, Error> uploadWithRoi(const std::byte* pSrc, const Extent& extent, const Roi& roi,
                                                           size_t bufferOffset, size_t bufferRowPitch) noexcept;
    [[nodiscard]] std::expected<void, Error> download(std::byte* pDst) noexcept;
    [[nodiscard]] std::expected<void, Error> downloadWithRoi(std::byte* pDst, const Extent& extent, const Roi& roi,
                                                             size_t bufferOffset, size_t bufferRowPitch) noexcept;

private:
    BufferBox bufferBox_;
    MemoryBox memoryBox_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/staging_buffer.cpp"
#endif
