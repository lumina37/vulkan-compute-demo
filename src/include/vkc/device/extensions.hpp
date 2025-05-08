#pragma once

#include <algorithm>
#include <expected>
#include <functional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "vkc/device/concepts.hpp"
#include "vkc/helper/error.hpp"

namespace vkc {

template <CHasExtensionName TVkProps>
static constexpr std::string_view extractName(const TVkProps& props) {
    return props.extensionName;
}

template <CHasLayerName TVkProps>
static constexpr std::string_view extractName(const TVkProps& props) {
    return props.layerName;
}

template <CExt TExt_>
class ExtEntry_ {
public:
    using TExt = TExt_;

private:
    ExtEntry_(std::string_view key, std::reference_wrapper<const TExt> extRef) noexcept;

public:
    [[nodiscard]] static ExtEntry_ fromExt(const TExt& ext) noexcept;

    friend constexpr auto operator<=>(const ExtEntry_& lhs, const ExtEntry_& rhs) noexcept {
        return lhs.getKey() <=> rhs.getKey();
    }
    friend constexpr auto operator==(const ExtEntry_& lhs, const ExtEntry_& rhs) noexcept {
        return lhs.getKey() == rhs.getKey();
    }

    template <typename Self>
    [[nodiscard]] auto&& getKey(this Self&& self) noexcept {
        return std::forward_like<Self>(self).key_;
    }

private:
    std::string_view key_;
    std::reference_wrapper<const TExt> extRef_;
};

template <CExt TExt_>
class ExtEntries_ {
public:
    using TExt = TExt_;
    using TEntry = ExtEntry_<TExt>;

private:
    ExtEntries_(std::vector<TExt>&& exts, std::vector<TEntry>&& extEntries) noexcept;

public:
    [[nodiscard]] static std::expected<ExtEntries_, Error> create(std::vector<TExt>&& exts) noexcept;

    [[nodiscard]] bool has(std::string_view key) const noexcept;

private:
    std::vector<TExt> exts_;
    std::vector<TEntry> extEntries_;
};

namespace rgs = std::ranges;

template <CExt TExt_>
ExtEntry_<TExt_>::ExtEntry_(std::string_view key, std::reference_wrapper<const TExt> extRef) noexcept
    : key_(key), extRef_(extRef) {}

template <CExt TExt_>
ExtEntry_<TExt_> ExtEntry_<TExt_>::fromExt(const TExt& ext) noexcept {
    const std::string_view key = extractName(ext);
    return ExtEntry_{key, std::cref(ext)};
}

template <CExt TExt_>
ExtEntries_<TExt_>::ExtEntries_(std::vector<TExt>&& exts, std::vector<TEntry>&& extEntries) noexcept
    : exts_(std::move(exts)), extEntries_(std::move(extEntries)) {}

template <CExt TExt_>
std::expected<ExtEntries_<TExt_>, Error> ExtEntries_<TExt_>::create(std::vector<TExt>&& exts) noexcept {
    auto extEntries = exts | rgs::views::transform(TEntry::fromExt) | rgs::to<std::vector>();
    rgs::sort(extEntries);
    return ExtEntries_{std::move(exts), std::move(extEntries)};
}

template <CExt TExt>
bool ExtEntries_<TExt>::has(std::string_view key) const noexcept {
    return rgs::binary_search(extEntries_, key, std::less{}, [](const TEntry& entry) { return entry.getKey(); });
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/extensions.cpp"
#endif
