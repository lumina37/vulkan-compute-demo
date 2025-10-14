#pragma once

#include <cstddef>
#include <span>

namespace shader::sgemm::dbg {

namespace wt0 {

extern const std::span<std::byte> code;

}  // namespace wt0

namespace wt1 {

extern const std::span<std::byte> code;

}  // namespace wt1

}  // namespace shader::sgemm::dbg
