#pragma once

#include <expected>
#include <print>
#include <utility>

#include "vkc/device/instance.hpp"
#include "vkc/device/physical/concepts.hpp"
#include "vkc/device/physical/manager.hpp"
#include "vkc/device/physical/props.hpp"
#include "vkc/device/score.hpp"
#include "vkc/helper/defines.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <CPhyDeviceProps TProps_>
class PhyDeviceSet_ {
public:
    using TProps = TProps_;
    using TPhyDeviceWithProps = PhyDeviceWithProps_<TProps>;
    using FnJudge = std::expected<float, Error> (*)(const TPhyDeviceWithProps&) noexcept;

private:
    PhyDeviceSet_(std::vector<TPhyDeviceWithProps>&& phyDevicesWithProps) noexcept;

public:
    [[nodiscard]] static std::expected<PhyDeviceSet_, Error> create(const InstanceManager& instMgr) noexcept;

    [[nodiscard]] std::expected<std::reference_wrapper<TPhyDeviceWithProps>, Error> select(
        const FnJudge& judge) noexcept;
    [[nodiscard]] std::expected<std::reference_wrapper<TPhyDeviceWithProps>, Error> selectDefault() noexcept;

private:
    std::vector<TPhyDeviceWithProps> phyDevicesWithProps_;
};

template <CPhyDeviceProps TProps>
PhyDeviceSet_<TProps>::PhyDeviceSet_(std::vector<TPhyDeviceWithProps>&& phyDevicesWithProps) noexcept
    : phyDevicesWithProps_(std::move(phyDevicesWithProps)) {}

template <CPhyDeviceProps TProps>
std::expected<PhyDeviceSet_<TProps>, Error> PhyDeviceSet_<TProps>::create(const InstanceManager& instMgr) noexcept {
    const auto& instance = instMgr.getInstance();

    const auto [physicalDevicesRes, physicalDevices] = instance.enumeratePhysicalDevices();
    if (physicalDevicesRes != vk::Result::eSuccess) {
        return std::unexpected{Error{physicalDevicesRes}};
    }

    std::vector<TPhyDeviceWithProps> phyDevicesWithProps;
    phyDevicesWithProps.reserve(physicalDevices.size());
    for (const auto& physicalDevice : physicalDevices) {
        auto phyDeviceMgrRes = PhyDeviceManager::create(physicalDevice);
        if (!phyDeviceMgrRes) return std::unexpected{std::move(phyDeviceMgrRes.error())};
        auto& phyDeviceMgr = phyDeviceMgrRes.value();

        auto phyDevicePropsRes = TProps::create(phyDeviceMgr);
        if (!phyDevicePropsRes) return std::unexpected{std::move(phyDevicePropsRes.error())};
        auto& phyDeviceProps = phyDevicePropsRes.value();

        phyDevicesWithProps.emplace_back(std::move(phyDeviceMgr), std::move(phyDeviceProps));
    }

    return PhyDeviceSet_{std::move(phyDevicesWithProps)};
}

template <CPhyDeviceProps TProps>
std::expected<std::reference_wrapper<PhyDeviceWithProps_<TProps>>, Error> PhyDeviceSet_<TProps>::select(
    const FnJudge& judge) noexcept {
    std::vector<Score<std::reference_wrapper<TPhyDeviceWithProps>>> scores;
    scores.reserve(phyDevicesWithProps_.size());

    const auto printDeviceInfo = [](const TPhyDeviceWithProps& deviceWithProps,
                                    const float score) -> std::expected<void, Error> {
        const auto phyDevice = deviceWithProps.getPhyDeviceMgr().getPhyDevice();
        const auto& phyDeviceProp = phyDevice.getProperties();
        const uint32_t apiVersion = deviceWithProps.getPhyDeviceProps().apiVersion;

        std::println("Candidate physical device: {}. Vk API version: {}.{}.{}. Score: {}",
                     phyDeviceProp.deviceName.data(), vk::apiVersionMajor(apiVersion), vk::apiVersionMinor(apiVersion),
                     vk::apiVersionPatch(apiVersion), score);

        return {};
    };

    for (auto& deviceWithProps : phyDevicesWithProps_) {
        auto scoreRes = judge(deviceWithProps);
        if (!scoreRes) return std::unexpected{std::move(scoreRes.error())};

        if constexpr (ENABLE_DEBUG) {
            auto printRes = printDeviceInfo(deviceWithProps, scoreRes.value());
            if (!printRes) return std::unexpected{std::move(printRes.error())};
        }

        scores.emplace_back(scoreRes.value(), std::ref(deviceWithProps));
    }

    if (scores.empty()) {
        return std::unexpected{Error{-1, "no sufficient device"}};
    }

    auto maxScoreIt = std::max_element(scores.begin(), scores.end());
    return std::move(maxScoreIt->attachment);
}

template <CPhyDeviceProps TProps>
std::expected<std::reference_wrapper<PhyDeviceWithProps_<TProps>>, Error>
PhyDeviceSet_<TProps>::selectDefault() noexcept {
    constexpr auto defaultJudge = [](const TPhyDeviceWithProps& phyDeviceWithProps) noexcept {
        return phyDeviceWithProps.getPhyDeviceProps().score();
    };
    return select(defaultJudge);
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/physical/set.cpp"
#endif
