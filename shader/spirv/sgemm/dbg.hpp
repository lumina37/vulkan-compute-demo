#pragma once

#include <cstddef>
#include <span>

namespace shader::sgemm::dbg {

namespace wt0 {

extern const std::span<std::byte> code;

}  // namespace wt0

namespace ggml {

extern const std::span<std::byte> code;

}  // namespace ggml

}  // namespace shader::sgemm::dbg
