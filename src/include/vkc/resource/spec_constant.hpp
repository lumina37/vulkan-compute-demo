#pragma once

#include <array>

#include "vkc/helper/std.hpp"
#include "vkc/helper/vulkan.hpp"

namespace vkc {

template <typename... TSc>
class SpecConstantBox {
public:
    constexpr SpecConstantBox(TSc... specConstants) noexcept;

    template <typename Self>
    [[nodiscard]] constexpr auto&& getSpecInfo(this Self&& self) noexcept {
        return std::forward_like<Self>(self).specInfo_;
    }

private:
    std::tuple<TSc...> specConstants_;

    std::array<vk::SpecializationMapEntry, sizeof...(TSc)> specMapEntries_;
    vk::SpecializationInfo specInfo_;
};

template <typename... TSc>
constexpr SpecConstantBox<TSc...>::SpecConstantBox(TSc... specConstants) noexcept : specConstants_(specConstants...) {
    const auto genSpecMapEntry = [&]<size_t index>() {
        const auto& specConstant = std::get<index>(specConstants_);
        const size_t offset = (size_t)&specConstant - (size_t)&specConstants_;

        vk::SpecializationMapEntry specMapEntry;
        specMapEntry.setConstantID((uint32_t)index);
        specMapEntry.setOffset((uint32_t)offset);
        specMapEntry.setSize(sizeof(specConstant));

        return specMapEntry;
    };

    const auto genSpecMapEnrtyHelper = [&]<size_t... Is>(std::index_sequence<Is...>) {
        return std::array { genSpecMapEntry.template operator()<Is>()... };
    };

    specMapEntries_ = genSpecMapEnrtyHelper(std::index_sequence_for<TSc...>{});

    specInfo_.setMapEntries(specMapEntries_);
    specInfo_.setDataSize(sizeof(specConstants_));
    specInfo_.setPData(&specConstants_);
}

}  // namespace vkc
