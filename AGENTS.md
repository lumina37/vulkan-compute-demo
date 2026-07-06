# vulkan-compute-demo 开发指南

本文档旨在帮助开发者快速了解项目结构，以便参与 vulkan-compute-demo 的开发

## 项目概述

vulkan-compute-demo 是一个使用 C++23 编写的 Vulkan 计算管线演示项目。该项目包含多种 GPU 计算算法（高斯模糊、SGEMM、Flash
Attention 2、前缀和、Top-K 等）的逐步优化实现，并封装了一套轻量级 RAII 风格的 Vulkan 计算辅助库（`vkc`）。

## 目录结构

```text
vulkan-compute-demo/
├── src/                              # vkc 封装库源代码
│   ├── CMakeLists.txt                # 构建配置（静态库 + header-only双模式）
│   └── include/vkc/                  # 公有头文件
│       ├── vkc.hpp                   # 总头文件
│       ├── command/                  # CommandPool 与 CommandBuffer 封装（命令录制、Pipeline Barrier）
│       ├── descriptor/               # Descriptor 封装（DescriptorSetLayout / Pool / Set）
│       ├── device/                   # 设备封装（Instance / 物理设备枚举选择 / 逻辑设备 / Queue Family）
│       ├── gui/                      # GLFW 窗口 + 交换链 + 表面封装
│       ├── helper/                   # 辅助工具（Error / 数学 / 平台宏 / Vulkan初始化）
│       ├── pipeline/                 # Pipeline 封装（PipelineLayout / Compute Pipeline）
│       ├── query_pool/               # QueryPool 封装（性能计数器 / 时间戳查询）
│       ├── resource/                 # 资源封装（Buffer / Image / 内存 / 各类专用缓冲区与图像）
│       ├── sync/                     # 同步原语封装（Fence / Semaphore）
│       ├── extent.hpp                # Extent 与 Roi 辅助类
│       ├── queue.hpp                 # QueueBox 计算队列封装
│       ├── shader.hpp                # ShaderBox 着色器模块封装
│       └── stb_image.hpp             # 图片 I/O 封装（stb_image）
├── shader/                           # 着色器源码与编译产物
│   ├── CMakeLists.txt                # 构建配置（调用compile_shaders.py）
│   ├── compile_shaders.py            # GLSL → SPIR-V 编译脚本
│   ├── shader.hpp                    # Shader总头文件
│   ├── glsl/<algorithm>/             # GLSL源码（.comp文件，按算法和版本组织）
│   └── spirv/<algorithm>/            # 编译生成的SPIR-V嵌入C头文件
├── samples/                          # 示例程序（每种算法变体一个可执行文件）
│   ├── CMakeLists.txt                # vkc_add_sample 注册所有示例
│   ├── vkc_helper.hpp               # 通用辅助（Unwrap / Timer / float16 / meanStd）
│   ├── gui.cpp / perf.cpp           # GUI窗口 / 性能查询演示
│   ├── gaussFilter/                  # 高斯模糊示例（baseline / separable / grayscale-after）
│   ├── grayscale/                    # 灰度转换示例（ROI / storage-image-input）
│   └── sgemm/                        # SGEMM示例
│       ├── simt/                     # SIMT普通计算路径优化阶梯
│       ├── tcore/                    # TensorCore加速路径优化阶梯
│       └── dbg/                      # 调试/验证变体
├── tests/                            # 单元测试（Catch2 v3）
│   ├── CMakeLists.txt                # vkc_add_test 注册所有测试
│   ├── vkc_helper.hpp                # 通用辅助（Unwrap / Timer / float16）
│   └── test_*.cpp                    # GPU vs CPU 正确性验证
├── CMakeLists.txt                    # 根构建文件（C++23, FetchContent依赖管理）
├── .clang-format                     # C++ 代码格式化配置
├── ruff.toml                         # Python 代码检查配置
├── LICENSE                           # MIT 许可证
└── README.md                         # 项目介绍
```

## 核心概念

### 术语定义

| 术语                     | 说明                                           |
|------------------------|----------------------------------------------|
| **Box**                | 资源封装类型的命名后缀，表示RAII管理的Vulkan对象                |
| **SC / SpecConstant**  | Shader Specialization Constant，CPU端传入的编译时常量  |
| **PC / PushConstant**  | Push Constant，命令录制时动态传入的小块常量数据               |
| **Tcore / TensorCore** | Tensor Core 加速路径（使用cooperative matrix）       |
| **SGEMM**              | 单精度矩阵乘                                       |
| **RCC**                | Row-major x Column-major = Column-major 矩阵乘法 |
| **RRR**                | Row-major x Row-major = Row-major 矩阵乘法       |

### 核心类与数据流

一个典型的GPU计算流程如下：

```text
main()
    │
    ├── vkc::initVulkan()              # 加载Vulkan动态库
    │
    ├── InstanceBox::create()          # 创建Vulkan实例（可指定扩展/验证层）
    ├── PhyDeviceSet_<Props>::create() # 枚举所有物理设备
    ├── selectDefault() / select(fn)   # 评分选择最优设备
    ├── PhyDeviceBox::create()         # 物理设备封装
    ├── DeviceBox::create()            # 逻辑设备（创建Queue）
    ├── QueueBox::create()             # 获取计算队列
    │
    ├── 资源创建:
    │   ├── StorageBufferBox::create() # 计算输入/输出缓冲区
    │   ├── StagingBufferBox::create() # 暂存缓冲区（CPU↔GPU传输）
    │   ├── SampledImageBox::create()  # 采样图像输入
    │   └── StorageImageBox::create()  # 存储图像输出
    │
    ├── 描述符设置:
    │   ├── DescPoolBox::create()
    │   ├── DescSetLayoutBox::create() # 布局由各资源的draftDescSetLayoutBinding()组合
    │   ├── PipelineLayoutBox::create()
    │   └── DescSetsBox::create()      # 写入由各资源的draftWriteDescSet()组合
    │
    ├── Pipeline:
    │   ├── ShaderBox::create(device, bytes)
    │   ├── SpecConstantBox{...}       # 指定workgroup size和矩阵维度
    │   └── PipelineBox::createCompute()
    │
    ├── 命令录制:
    │   ├── CommandPoolBox::create()
    │   ├── CommandBufferBox::create()
    │   ├── recordBegin() → bindPipeline → bindDescSets
    │   ├── recordPrepareReceive() / recordPrepareShaderRead()  # Pipeline Barrier
    │   ├── recordCopyStagingToBuffer() / recordCopyStagingToImage() # 数据上传
    │   ├── recordDispatch()           # 启动计算
    │   ├── recordPrepareSend()        # Pipeline Barrier（GPU→传输）
    │   └── recordCopyBufferToStaging() # 结果下载
    │
    ├── QueueBox::submit(waitFence)    # 提交到计算队列
    ├── FenceBox::wait()               # 等待完成
    │
    └── StagingBufferBox::read()       # 读取结果到CPU内存
```

### 关键设计模式

1. **[[nodiscard]] + std::expected**：所有可能失败的函数返回 `std::expected<T, Error>`，强制处理错误
2. **RAII Box**：每个Vulkan对象封装为 move-only 的 `*Box` 类，析构时自动释放资源
3. **共享Device所有权**：派生资源持有 `std::shared_ptr<DeviceBox>`，无论资源销毁顺序如何，Device始终最后析构
4. **工厂方法 + 私有构造**：所有Box类通过 `static std::expected<T, Error> create(...)` 创建，构造函数为private
5. **deducing this**：使用C++23 `Self`模式（`template <typename Self> auto getXxx(this Self&& self)`
   ）实现getter的const/非const转发
6. **Header-only兼容**：每个 `.hpp` 末尾有 `#ifdef _VKC_LIB_HEADER_ONLY #include "...cpp" #endif`，通过宏切换静态库/header-only模式
7. **Concept约束**：`CImageBox`、`CBufferBox` 等 Concept 用于编译期检查传入类型的合法性

## Shader模块规范

- GLSL源码位于 `shader/glsl/<algorithm>/`，编译产物位于 `shader/spirv/<algorithm>/`
- SPIR-V二进制以 `uint32_t code[]` 嵌入C头文件，通过 `shader::<algorithm>::<variant>::code`（数据类型为
  `std::span<std::byte>`）访问
- shader内容发生变化时自动运行 `shader/compile_shaders.py` 编译发生变化的shader

## Sample模块规范

- 每个shader变体对应一个示例，放在 `samples/<algorithm>/<variant>.cpp`
- 通过 `samples/CMakeLists.txt` 中的 `vkc_add_sample(name src)` 注册

## 测试模块规范

- 使用Catch2 v3，测试文件位于 `tests/test_*.cpp`
- 每个测试实现CPU参考算法与GPU结果对比，验证正确性
- 使用 `SECTION()` 组织同一算法的多个变体测试
- 通过 `tests/CMakeLists.txt` 中的 `vkc_add_test(name lib srcs)` 注册

## 开发规范

参阅 `.github/CONTRIBUTING.md`
