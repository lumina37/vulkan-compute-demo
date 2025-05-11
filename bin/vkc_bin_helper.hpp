#pragma once

#include <expected>
#include <filesystem>
#include <iostream>
#include <print>
#include <string>
#include <type_traits>
#include <utility>

#include "vkc.hpp"

namespace fs = std::filesystem;

class Unwrap {
public:
    template <typename T>
    static friend auto operator|(std::expected<T, vkc::Error>&& src, [[maybe_unused]] const Unwrap& _) {
        if (!src.has_value()) {
            const auto& err = src.error();
            const fs::path filePath{err.source.file_name()};
            const std::string fileName = filePath.filename().string();
            std::println(std::cerr, "{}:{} msg={} code={}", fileName, err.source.line(), err.msg, err.code);
            std::exit(err.code);
        }
        if constexpr (!std::is_void_v<T>) {
            return std::forward_like<T>(src.value());
        }
    }
};

constexpr auto unwrap = Unwrap();
