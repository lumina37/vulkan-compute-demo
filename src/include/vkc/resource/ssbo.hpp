#pragma once

#include <vulkan/vulkan.hpp>

#include "vkc/device/logical.hpp"
#include "vkc/device/physical.hpp"

namespace vkc {

class SSBOManager {
public:
    SSBOManager(const PhyDeviceManager& phyDeviceMgr, DeviceManager& deviceMgr, vk::DeviceSize size);
    ~SSBOManager() noexcept;

    [[nodiscard]] vk::DeviceSize getSize() const noexcept { return size_; }

    template <typename Self>
    [[nodiscard]] auto&& getMemory(this Self&& self) noexcept {
        return std::forward_like<Self>(self).memory_;
    }

    template <typename Self>
    [[nodiscard]] auto&& getBuffer(this Self&& self) noexcept {
        return std::forward_like<Self>(self).buffer_;
    }

    [[nodiscard]] constexpr vk::DescriptorType getDescType() const noexcept {
        return vk::DescriptorType::eStorageBuffer;
    }
    [[nodiscard]] vk::WriteDescriptorSet draftWriteDescSet() const noexcept;

    vk::Result uploadFrom(std::span<std::byte> data);
    vk::Result downloadTo(std::span<std::byte> data);

private:
    DeviceManager& deviceMgr_;  // FIXME: UAF
    vk::DeviceSize size_;
    vk::DeviceMemory memory_;
    vk::Buffer buffer_;
    vk::DescriptorBufferInfo bufferInfo_;
};

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/image.cpp"
#endif
