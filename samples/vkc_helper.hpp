#pragma once

#include <chrono>
#include <cstdint>
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
    friend auto operator|(std::expected<T, vkc::Error>&& src, [[maybe_unused]] const Unwrap& _) {
        if (!src.has_value()) {
            const auto& err = src.error();
            const fs::path filePath{err.source.file_name()};
            const std::string fileName = filePath.filename().string();
            std::println(std::cerr, "{}:{} cate={} code={} msg={}", fileName, err.source.line(),
                         vkc::errCateToStr(err.cate), err.code, err.msg);
            std::exit(err.code);
        }
        if constexpr (!std::is_void_v<T>) {
            return std::forward_like<T>(src.value());
        }
    }
};

constexpr auto unwrap = Unwrap();

class Timer {
public:
    void begin() noexcept { begin_ = std::chrono::steady_clock::now(); }
    void end() noexcept {
        const auto end = std::chrono::steady_clock::now();
        durationNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin_).count();
    }
    [[nodiscard]] float durationMs() const noexcept { return (float)durationNs_ / 1000000.0f; }

private:
    std::chrono::time_point<std::chrono::steady_clock> begin_;
    int64_t durationNs_{};
};

class float16 {
public:
    float16() : bits_(0) {}
    float16(float value) : bits_(float32ToFloat16(value)) {}

    operator float() const { return float16ToFloat32(bits_); }

    float16& operator=(float value) {
        bits_ = float32ToFloat16(value);
        return *this;
    }

private:
    uint16_t bits_;

    static uint16_t float32ToFloat16(float value) {
        uint32_t f32 = *reinterpret_cast<uint32_t*>(&value);

        uint32_t sign = (f32 >> 16) & 0x8000;
        int32_t exponent = ((f32 >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = f32 & 0x007FFFFF;

        if (((f32 >> 23) & 0xFF) == 0xFF) {
            if (mantissa != 0)
                return sign | 0x7E00;  // NaN
            else
                return sign | 0x7C00;  // Inf
        }

        if (exponent <= 0) {
            if (exponent < -10) return static_cast<uint16_t>(sign);
            mantissa = (mantissa | 0x00800000) >> (1 - exponent);
            if (mantissa & 0x00001000) mantissa += 0x00002000;
            return static_cast<uint16_t>(sign | (mantissa >> 13));
        }

        if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7C00);

        if (mantissa & 0x00001000) mantissa += 0x00002000;

        uint16_t result = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
        return result;
    }

    static float float16ToFloat32(uint16_t bits) {
        uint32_t sign = (bits & 0x8000) << 16;
        uint32_t exponent = (bits & 0x7C00) >> 10;
        uint32_t mantissa = (bits & 0x03FF);

        uint32_t f32;
        if (exponent == 0) {
            if (mantissa == 0) {
                f32 = sign;
            } else {
                exponent = 1;
                while ((mantissa & 0x0400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x03FF;
                exponent = exponent + (127 - 15);
                mantissa <<= 13;
                f32 = sign | (exponent << 23) | mantissa;
            }
        } else if (exponent == 0x1F) {
            f32 = sign | 0x7F800000 | (mantissa << 13);
        } else {
            exponent = exponent + (127 - 15);
            mantissa <<= 13;
            f32 = sign | (exponent << 23) | mantissa;
        }

        return *reinterpret_cast<float*>(&f32);
    }
};
