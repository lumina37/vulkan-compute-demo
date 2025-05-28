#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "vkc/device/logical.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc::_hp {

[[nodiscard]] std::expected<uint32_t, Error> findMemoryTypeIdx(const PhyDeviceManager& phyDeviceMgr,
                                                               uint32_t supportedMemType,
                                                               vk::MemoryPropertyFlags memProps) noexcept;

[[nodiscard]] std::expected<void, Error> allocBufferMemory(const PhyDeviceManager& phyDeviceMgr,
                                                           DeviceManager& deviceMgr, vk::Buffer& buffer,
                                                           vk::MemoryPropertyFlags memProps,
                                                           vk::DeviceMemory& bufferMemory) noexcept;

[[nodiscard]] std::expected<void, Error> allocImageMemory(const PhyDeviceManager& phyDeviceMgr,
                                                          DeviceManager& deviceMgr, vk::Image& image,
                                                          vk::MemoryPropertyFlags memProps,
                                                          vk::DeviceMemory& bufferMemory) noexcept;

[[nodiscard]] std::expected<void, Error> uploadFrom(DeviceManager& deviceMgr, vk::DeviceMemory& memory,
                                                    std::span<const std::byte> data) noexcept;

[[nodiscard]] std::expected<void, Error> downloadTo(DeviceManager& deviceMgr, const vk::DeviceMemory& memory,
                                                    std::span<std::byte> data) noexcept;

class MemMapManager {
    MemMapManager(std::shared_ptr<DeviceManager>&& pDeviceMgr, vk::DeviceMemory memory, void* mapPtr) noexcept;

public:
    MemMapManager(MemMapManager&& rhs) noexcept;
    ~MemMapManager() noexcept;

    [[nodiscard]] static std::expected<MemMapManager, Error> create(std::shared_ptr<DeviceManager> pDeviceMgr,
                                                                    vk::DeviceMemory& memory, size_t size) noexcept;

    [[nodiscard]] void* getMapPtr() const noexcept { return mapPtr_; }

private:
    std::shared_ptr<DeviceManager> pDeviceMgr_;

    vk::DeviceMemory memory_;
    void* mapPtr_;
};

}  // namespace vkc::_hp

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/resource/memory.cpp"
#endif
