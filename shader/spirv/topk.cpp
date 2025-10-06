#include <cstddef>
#include <span>

#include "spirv/topk.hpp"

namespace shader::topk {

namespace v0 {

namespace _detail {
#include "spirv/topk/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

}  // namespace shader::topk
