#pragma once

#include <expected>
#include <print>
#include <utility>

#include "vkc/device/concepts.hpp"
#include "vkc/device/instance.hpp"
#include "vkc/device/physical.hpp"
#include "vkc/device/props.hpp"
#include "vkc/device/score.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

std::expected<float, Error> defaultJudge(const PhyDeviceWithProps_<PhyDeviceProps>& phyDeviceWithProps) noexcept;

template <CPhyDeviceProps TProps_>
class DeviceSet_ {
public:
    using TProps = TProps_;
    using TDeviceWithProps = PhyDeviceWithProps_<TProps>;
    using FnJudge = std::expected<float, Error> (*)(const TDeviceWithProps&) noexcept;

private:
    DeviceSet_(std::vector<TDeviceWithProps>&& deviceWithPropsVec) noexcept;

public:
    [[nodiscard]] static std::expected<DeviceSet_, Error> create(const InstanceManager& instMgr) noexcept;

    [[nodiscard]] std::expected<std::reference_wrapper<TDeviceWithProps>, Error> select(const FnJudge& judge) noexcept;
    [[nodiscard]] std::expected<std::reference_wrapper<TDeviceWithProps>, Error> pickDefault() noexcept;

private:
    std::vector<TDeviceWithProps> deviceWithPropsVec_;
};

template <CPhyDeviceProps TProps>
DeviceSet_<TProps>::DeviceSet_(std::vector<TDeviceWithProps>&& deviceWithPropsVec) noexcept
    : deviceWithPropsVec_(std::move(deviceWithPropsVec)) {}

template <CPhyDeviceProps TProps>
std::expected<DeviceSet_<TProps>, Error> DeviceSet_<TProps>::create(const InstanceManager& instMgr) noexcept {
    const auto& instance = instMgr.getInstance();

    const auto [physicalDevicesRes, physicalDevices] = instance.enumeratePhysicalDevices();
    if (physicalDevicesRes != vk::Result::eSuccess) {
        return std::unexpected{Error{physicalDevicesRes}};
    }

    std::vector<TDeviceWithProps> deviceWithPropsVec;
    deviceWithPropsVec.reserve(physicalDevices.size());
    for (const auto& physicalDevice : physicalDevices) {
        auto phyDeviceMgrRes = PhyDeviceManager::create(physicalDevice);
        if (!phyDeviceMgrRes) return std::unexpected{std::move(phyDeviceMgrRes.error())};
        auto& phyDeviceMgr = phyDeviceMgrRes.value();

        auto phyDevicePropsRes = TProps::create(phyDeviceMgr);
        if (!phyDevicePropsRes) return std::unexpected{std::move(phyDevicePropsRes.error())};
        auto& phyDeviceProps = phyDevicePropsRes.value();

        deviceWithPropsVec.emplace_back(std::move(phyDeviceMgr), std::move(phyDeviceProps));
    }

    return DeviceSet_{std::move(deviceWithPropsVec)};
}

template <CPhyDeviceProps TProps>
std::expected<std::reference_wrapper<PhyDeviceWithProps_<TProps>>, Error> DeviceSet_<TProps>::select(
    const FnJudge& judge) noexcept {
    std::vector<Score<std::reference_wrapper<TDeviceWithProps>>> scores;
    scores.reserve(deviceWithPropsVec_.size());

    const auto printDeviceInfo = [](const TDeviceWithProps& deviceWithProps,
                                    const float score) -> std::expected<void, Error> {
        const auto rstrip = [](std::string_view str) {
            size_t lastCh = str.find_last_not_of(' ');
            return str.substr(0, lastCh + 1);
        };

        const auto phyDevice = deviceWithProps.getPhyDeviceMgr().getPhyDevice();
        const auto& phyDeviceProp = phyDevice.getProperties();
        const uint32_t apiVersion = deviceWithProps.getProps().apiVersion;

        std::println("Candidate physical device: {}. Vk API version: {}.{}.{}. Score: {}",
                     phyDeviceProp.deviceName.data(), VK_API_VERSION_MAJOR(apiVersion),
                     VK_API_VERSION_MINOR(apiVersion), VK_API_VERSION_PATCH(apiVersion), score);

        return {};
    };

    for (auto& deviceWithProps : deviceWithPropsVec_) {
        auto scoreRes = judge(deviceWithProps);
        if (!scoreRes) return std::unexpected{std::move(scoreRes.error())};

        if constexpr (ENABLE_DEBUG) {
            auto printRes = printDeviceInfo(deviceWithProps, scoreRes.value());
            if (!printRes) return std::unexpected{std::move(printRes.error())};
        }

        scores.emplace_back(scoreRes.value(), std::ref(deviceWithProps));
    }

    if (scores.empty()) {
        return std::unexpected{Error{1, "no sufficient device"}};
    }

    auto maxScoreIt = std::max_element(scores.begin(), scores.end());
    return std::move(maxScoreIt->attachment);
}

template <CPhyDeviceProps TProps>
std::expected<std::reference_wrapper<PhyDeviceWithProps_<TProps>>, Error> DeviceSet_<TProps>::pickDefault() noexcept {
    return select(defaultJudge);
}

}  // namespace vkc

#ifdef _vkc_LIB_HEADER_ONLY
#    include "vkc/device/set.cpp"
#endif
