#pragma once

#include <algorithm>
#include <functional>
#include <ranges>
#include <string>
#include <vector>

#include "vkc/device/concepts.hpp"
#include "vkc/helper/error.hpp"
#include "vkc/helper/std.hpp"

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
    [[nodiscard]] static ExtEntry_ createWithoutErr(const TExt& ext) noexcept;

    friend constexpr auto operator<=>(const ExtEntry_& lhs, const ExtEntry_& rhs) noexcept {
        return lhs.getKey() <=> rhs.getKey();
    }
    friend constexpr auto operator==(const ExtEntry_& lhs, const ExtEntry_& rhs) noexcept {
        return lhs.getKey() == rhs.getKey();
    }

    [[nodiscard]] std::string_view getKey() const noexcept { return key_; }
    [[nodiscard]] static std::string_view exposeKey(const ExtEntry_& entry) noexcept { return entry.getKey(); }

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
    ExtEntries_() noexcept = default;

    [[nodiscard]] static std::expected<ExtEntries_, Error> create(std::vector<TExt>&& exts) noexcept;

    [[nodiscard]] bool has(std::string_view key) const noexcept;

private:
    std::vector<TExt> exts_;
    std::vector<TEntry> extEntries_;
};

namespace rgs = std::ranges;

template <CExt TExt>
ExtEntry_<TExt>::ExtEntry_(std::string_view key, std::reference_wrapper<const TExt> extRef) noexcept
    : key_(key), extRef_(extRef) {}

template <CExt TExt>
ExtEntry_<TExt> ExtEntry_<TExt>::createWithoutErr(const TExt& ext) noexcept {
    const std::string_view key = extractName(ext);
    return ExtEntry_{key, std::cref(ext)};
}

template <CExt TExt>
ExtEntries_<TExt>::ExtEntries_(std::vector<TExt>&& exts, std::vector<TEntry>&& extEntries) noexcept
    : exts_(std::move(exts)), extEntries_(std::move(extEntries)) {}

template <CExt TExt>
auto ExtEntries_<TExt>::create(std::vector<TExt>&& exts) noexcept -> std::expected<ExtEntries_, Error> {
    auto extEntries = exts | rgs::views::transform(TEntry::createWithoutErr) | rgs::to<std::vector>();
    rgs::sort(extEntries);
    return ExtEntries_{std::move(exts), std::move(extEntries)};
}

template <CExt TExt>
bool ExtEntries_<TExt>::has(std::string_view key) const noexcept {
    return rgs::binary_search(extEntries_, key, std::less{}, TEntry::exposeKey);
}

}  // namespace vkc

#ifdef _VKC_LIB_HEADER_ONLY
#    include "vkc/device/extensions.cpp"
#endif
