#pragma once

#include <cstddef>
#include <span>

namespace shader::grayscale {

namespace ro {

extern const std::span<std::byte> code;

}  // namespace ro

namespace rw {

extern const std::span<std::byte> code;

}  // namespace rw

}  // namespace shader::grayscale
