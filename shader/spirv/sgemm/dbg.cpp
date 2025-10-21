#include <cstddef>
#include <span>

#include "spirv/sgemm/dbg.hpp"

namespace shader::sgemm::dbg {

namespace simon {

namespace _detail {
#include "spirv/sgemm/dbg/simon.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace simon

namespace ggml {

namespace _detail {
#include "spirv/sgemm/dbg/ggml.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace ggml

namespace v0 {

namespace _detail {
#include "spirv/sgemm/dbg/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

}  // namespace shader::sgemm::dbg
