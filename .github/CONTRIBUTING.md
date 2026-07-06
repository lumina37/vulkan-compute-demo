# 开发规范

## commit规范

提交信息须遵循简化版的[Conventional Commits](https://www.conventionalcommits.org/zh-hans/)，格式为 `<type>: <description>`。

| 类型 | 说明 |
| ------ | ------ |
| `feat` | 新功能 |
| `fix` | Bug修复 |
| `refactor` | 重构（既非新功能也非修复） |
| `perf` | 性能优化 |
| `chore` | 日常维护（依赖更新、脚本改进等） |
| `docs` | 文档变更 |
| `test` | 测试相关 |
| `style` | 代码格式（不影响逻辑的空白、缩进等） |
| `ci` | CI/CD 配置变更 |

## 代码风格

- C++代码风格遵循`.clang-format`（BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 120）
- Python代码风格遵循`ruff.toml`

## 命名约定

| 类型 | 约定 | 示例 |
| ------ | ------ | ------ |
| 类名 | PascalCase，资源封装类以 `Box` 结尾 | `DeviceBox`, `BufferBox` |
| 枚举名 | PascalCase，`E` 前缀 | `ECode`, `ECate` |
| 枚举值 | `e` 前缀 + PascalCase | `eUnknown`, `eResourceInvalid` |
| 函数/方法 | camelCase | `createCompute`, `getMemoryRequirements` |
| 成员变量 | camelCase + 尾部下划线 `_` | `device_`, `buffer_` |
| 局部变量/参数 | camelCase | `deviceBox`, `shaderBox` |
| 文件/目录名 | snake_case | `command_buffer.hpp`, `staging_buffer.cpp` |
| 命名空间 | 小写 + 简短 | `vkc`, `shader` |
| 模板参数 | PascalCase + 尾部下划线 `_` | `TFeat`, `TPc_` |
| Concept名 | `C` 前缀 + PascalCase | `CImageBox`, `CBufferBox` |
| 类型别名（Ref包装） | `T` 前缀 | `TStorageImageBoxRef` |
| 全局常量 | PascalCase（由命名空间/类名区分） | `shader::sgemm::simt::v0::code` |

## 错误处理规范

- 项目设置为 `VULKAN_HPP_NO_EXCEPTIONS`，禁止使用C++异常
- 所有可能失败的函数返回 `std::expected<T, Error>`，并以 `[[nodiscard]]` 标记
- `Error` 类位于 `vkc/helper/error.hpp`，包含错误分类（`ECate`）、错误码（`ECode`）、`source_location` 和可选消息
- 在samples中使用 `unwrap` pipe操作符提取 `expected` 值，失败时打印错误并退出

## 资源管理规范

- 所有Vulkan对象封装为RAII的 `*Box` 类，Non-copyable, Move-only
- 派生资源（Buffer/Image/DescriptorSet等）通过 `std::shared_ptr<DeviceBox>` 持有Device的共享所有权，确保Device在所有资源销毁后才析构
- 每个 `*Box` 类提供静态 `create(...) -> std::expected<T, Error>` 工厂方法，构造函数设为私有
- 使用 `deducing this`（C++23 `Self`模式）实现 `getVkXxx()` 转发方法

## Shader开发规范

- 计算着色器使用GLSL `#version 460`，文件后缀为 `.comp`
- 工作组分箱大小和矩阵维度使用specialization constant（`layout (constant_id = N)`）由CPU端传入
- Shader文件放在 `shader/glsl/<algorithm>/<variant>.comp`，对应的SPIR-V C头文件自动生成到 `shader/spirv/<algorithm>/<variant>.h`
- 新增/修改shader后运行 `shader/compile_shaders.py` 重新编译生成SPIR-V头文件
- SPIR-V代码以 `const std::span<std::byte> code` 变量暴露在 `shader::<algorithm>::<variant>` 命名空间下

## 文档更新规范

- 外部API变更时需同步更新 `README.md`

## 测试编写规范

- 使用Catch2 v3框架，测试文件放在 `tests/` 目录
- 测试需实现CPU参考算法并与GPU计算结果对比验证
- 使用 `SECTION()` 组织同一算法的多个shader变体测试
- 测试模板：`vkc_add_test(test_name vkc::lib::static "test_xxx.cpp")`

## 新增Shader/算法时的检查清单

- [ ] 在 `shader/glsl/<algorithm>/` 下创建 `.comp` 着色器文件
- [ ] 运行 `shader/compile_shaders.py` 生成SPIR-V头文件
- [ ] 在 `shader/spirv/` 下创建对应的 `.hpp` / `.cpp` 文件暴露 `code` 变量
- [ ] 在 `samples/` 下创建示例程序并通过 `CMakeLists.txt` 注册
- [ ] 如有测试，在 `tests/` 下创建测试文件并注册到 `tests/CMakeLists.txt`

## 新增Vulkan封装时的检查清单

- [ ] 在 `src/include/vkc/<category>/` 下创建 `.hpp` 头文件
- [ ] 如需编译实现，在相同目录创建 `.cpp` 文件
- [ ] 在头文件末尾添加 `#ifdef _VKC_LIB_HEADER_ONLY` 的header-only兼容块
- [ ] 返回类型使用 `std::expected<T, Error>` 并标记 `[[nodiscard]]`
- [ ] 确保资源正确释放（RAII析构函数）
- [ ] 运行 `cmake --build .` 确保编译通过
