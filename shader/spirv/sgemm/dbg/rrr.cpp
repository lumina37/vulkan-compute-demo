#include <cstddef>
#include <span>

#include "spirv/sgemm/dbg/rrr.hpp"

namespace shader::sgemm::dbg::rrr {

namespace simon {

namespace _detail {
#include "spirv/sgemm/dbg/rrr/simon.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace simon

namespace v0 {

namespace _detail {
#include "spirv/sgemm/dbg/rrr/v0.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v0

namespace v1 {

namespace _detail {
#include "spirv/sgemm/dbg/rrr/v1.h"
}

const std::span<std::byte> code{(std::byte*)_detail::code, sizeof(_detail::code)};

}  // namespace v1

}  // namespace shader::sgemm::dbg::rrr
