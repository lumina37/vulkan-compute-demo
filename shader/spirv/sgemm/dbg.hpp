#pragma once

#include <cstddef>
#include <span>

namespace shader::sgemm::dbg {

namespace simon {

extern const std::span<std::byte> code;

}  // namespace simon

namespace ggml {

extern const std::span<std::byte> code;

}  // namespace ggml

namespace v0 {

extern const std::span<std::byte> code;

}  // namespace v0

namespace v1 {

extern const std::span<std::byte> code;

}  // namespace v1

}  // namespace shader::sgemm::dbg
