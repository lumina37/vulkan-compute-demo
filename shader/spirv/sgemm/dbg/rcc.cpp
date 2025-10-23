#include <cstddef>
#include <span>

#include "spirv/sgemm/dbg/rcc.hpp"

namespace shader::sgemm::dbg::rcc {

namespace ggml {

namespace _detail {
#include "spirv/sgemm/dbg/rcc/ggml.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace ggml

namespace v0 {

namespace _detail {
#include "spirv/sgemm/dbg/rcc/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

}  // namespace shader::sgemm::dbg::rcc
