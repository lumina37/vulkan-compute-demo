#if 0
; SPIR-V
; Version: 1.0
; Generator: Google spiregg; 0
; Bound: 119
; Schema: 0
               OpCapability Shader
               OpCapability ImageQuery
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
               OpExecutionMode %main LocalSize 16 16 1
               OpSource HLSL 600
               OpName %type_ConstantBuffer_PushConstants "type.ConstantBuffer.PushConstants"
               OpMemberName %type_ConstantBuffer_PushConstants 0 "kernelSize"
               OpMemberName %type_ConstantBuffer_PushConstants 1 "sigma"
               OpName %pc "pc"
               OpName %type_2d_image "type.2d.image"
               OpName %srcTex "srcTex"
               OpName %type_sampler "type.sampler"
               OpName %srcSampler "srcSampler"
               OpName %type_2d_image_0 "type.2d.image"
               OpName %dstImage "dstImage"
               OpName %main "main"
               OpName %type_sampled_image "type.sampled.image"
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %srcTex DescriptorSet 0
               OpDecorate %srcTex Binding 0
               OpDecorate %srcSampler DescriptorSet 0
               OpDecorate %srcSampler Binding 1
               OpDecorate %dstImage DescriptorSet 0
               OpDecorate %dstImage Binding 2
               OpMemberDecorate %type_ConstantBuffer_PushConstants 0 Offset 0
               OpMemberDecorate %type_ConstantBuffer_PushConstants 1 Offset 4
               OpDecorate %type_ConstantBuffer_PushConstants Block
               OpDecorate %13 NoContraction
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %int_2 = OpConstant %int 2
      %float = OpTypeFloat 32
    %float_0 = OpConstant %float 0
    %v4float = OpTypeVector %float 4
         %23 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
  %float_0_5 = OpConstant %float 0.5
    %v2float = OpTypeVector %float 2
         %26 = OpConstantComposite %v2float %float_0_5 %float_0_5
    %float_2 = OpConstant %float 2
    %float_1 = OpConstant %float 1
%type_ConstantBuffer_PushConstants = OpTypeStruct %int %float
%_ptr_PushConstant_type_ConstantBuffer_PushConstants = OpTypePointer PushConstant %type_ConstantBuffer_PushConstants
%type_2d_image = OpTypeImage %float 2D 2 0 0 1 Unknown
%_ptr_UniformConstant_type_2d_image = OpTypePointer UniformConstant %type_2d_image
%type_sampler = OpTypeSampler
%_ptr_UniformConstant_type_sampler = OpTypePointer UniformConstant %type_sampler
%type_2d_image_0 = OpTypeImage %float 2D 2 0 0 2 Rgba8
%_ptr_UniformConstant_type_2d_image_0 = OpTypePointer UniformConstant %type_2d_image_0
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %37 = OpTypeFunction %void
      %v2int = OpTypeVector %int 2
     %v2uint = OpTypeVector %uint 2
%_ptr_PushConstant_int = OpTypePointer PushConstant %int
%type_sampled_image = OpTypeSampledImage %type_2d_image
%_ptr_PushConstant_float = OpTypePointer PushConstant %float
         %pc = OpVariable %_ptr_PushConstant_type_ConstantBuffer_PushConstants PushConstant
     %srcTex = OpVariable %_ptr_UniformConstant_type_2d_image UniformConstant
 %srcSampler = OpVariable %_ptr_UniformConstant_type_sampler UniformConstant
   %dstImage = OpVariable %_ptr_UniformConstant_type_2d_image_0 UniformConstant
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
     %uint_0 = OpConstant %uint 0
     %int_n2 = OpConstant %int -2
       %main = OpFunction %void None %37
         %44 = OpLabel
         %45 = OpLoad %v3uint %gl_GlobalInvocationID
               OpSelectionMerge %46 None
               OpSwitch %uint_0 %47
         %47 = OpLabel
         %48 = OpVectorShuffle %v2uint %45 %45 0 1
         %49 = OpBitcast %v2int %48
         %50 = OpLoad %type_2d_image_0 %dstImage
         %51 = OpImageQuerySize %v2uint %50
         %52 = OpCompositeExtract %uint %51 0
         %53 = OpBitcast %int %52
         %54 = OpCompositeExtract %uint %51 1
         %55 = OpBitcast %int %54
         %56 = OpCompositeConstruct %v2int %53 %55
         %57 = OpCompositeExtract %int %49 0
         %58 = OpSGreaterThanEqual %bool %57 %53
         %59 = OpLogicalNot %bool %58
               OpSelectionMerge %60 None
               OpBranchConditional %59 %61 %60
         %61 = OpLabel
         %62 = OpCompositeExtract %int %49 1
         %63 = OpSGreaterThanEqual %bool %62 %55
               OpBranch %60
         %60 = OpLabel
         %64 = OpPhi %bool %true %47 %63 %61
               OpSelectionMerge %65 None
               OpBranchConditional %64 %66 %65
         %66 = OpLabel
               OpBranch %46
         %65 = OpLabel
         %67 = OpAccessChain %_ptr_PushConstant_int %pc %int_0
         %68 = OpLoad %int %67
         %69 = OpSDiv %int %68 %int_2
         %70 = OpSDiv %int %68 %int_n2
               OpBranch %71
         %71 = OpLabel
         %72 = OpPhi %v4float %23 %65 %73 %74
         %75 = OpPhi %float %float_0 %65 %76 %74
         %77 = OpPhi %int %70 %65 %78 %74
         %79 = OpSLessThanEqual %bool %77 %69
               OpLoopMerge %80 %74 None
               OpBranchConditional %79 %81 %80
         %81 = OpLabel
               OpBranch %82
         %82 = OpLabel
         %76 = OpPhi %float %75 %81 %83 %84
         %73 = OpPhi %v4float %72 %81 %13 %84
         %85 = OpPhi %int %70 %81 %86 %84
         %87 = OpSLessThanEqual %bool %85 %69
               OpLoopMerge %88 %84 None
               OpBranchConditional %87 %84 %88
         %84 = OpLabel
         %89 = OpCompositeConstruct %v2int %85 %77
         %90 = OpIAdd %v2int %49 %89
         %91 = OpConvertSToF %v2float %90
         %92 = OpFAdd %v2float %91 %26
         %93 = OpConvertSToF %v2float %56
         %94 = OpFDiv %v2float %92 %93
         %95 = OpLoad %type_2d_image %srcTex
         %96 = OpLoad %type_sampler %srcSampler
         %97 = OpSampledImage %type_sampled_image %95 %96
         %98 = OpImageSampleExplicitLod %v4float %97 %94 Lod %float_0
         %99 = OpIMul %int %85 %85
        %100 = OpIMul %int %77 %77
        %101 = OpIAdd %int %99 %100
        %102 = OpConvertSToF %float %101
        %103 = OpFNegate %float %102
        %104 = OpAccessChain %_ptr_PushConstant_float %pc %int_1
        %105 = OpLoad %float %104
        %106 = OpFMul %float %105 %105
        %107 = OpFMul %float %106 %float_2
        %108 = OpFDiv %float %103 %107
        %109 = OpExtInst %float %1 Exp %108
        %110 = OpCompositeConstruct %v4float %109 %109 %109 %109
         %13 = OpExtInst %v4float %1 Fma %98 %110 %73
         %83 = OpFAdd %float %76 %109
         %86 = OpIAdd %int %85 %int_1
               OpBranch %82
         %88 = OpLabel
               OpBranch %74
         %74 = OpLabel
         %78 = OpIAdd %int %77 %int_1
               OpBranch %71
         %80 = OpLabel
        %111 = OpCompositeConstruct %v4float %75 %75 %75 %75
        %112 = OpFDiv %v4float %72 %111
        %113 = OpCompositeExtract %float %112 0
        %114 = OpCompositeExtract %float %112 1
        %115 = OpCompositeExtract %float %112 2
        %116 = OpCompositeConstruct %v4float %113 %114 %115 %float_1
        %117 = OpBitcast %v2uint %49
        %118 = OpLoad %type_2d_image_0 %dstImage
               OpImageWrite %118 %117 %116 None
               OpBranch %46
         %46 = OpLabel
               OpReturn
               OpFunctionEnd

#endif

const unsigned char g_main[] = {
  0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x77, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x32, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x06, 0x00, 0x01, 0x00, 0x00, 0x00, 0x47, 0x4c, 0x53, 0x4c,
  0x2e, 0x73, 0x74, 0x64, 0x2e, 0x34, 0x35, 0x30, 0x00, 0x00, 0x00, 0x00,
  0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x0f, 0x00, 0x06, 0x00, 0x05, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x06, 0x00, 0x02, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x03, 0x00, 0x05, 0x00, 0x00, 0x00, 0x58, 0x02, 0x00, 0x00,
  0x05, 0x00, 0x0b, 0x00, 0x04, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65,
  0x2e, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x42, 0x75, 0x66,
  0x66, 0x65, 0x72, 0x2e, 0x50, 0x75, 0x73, 0x68, 0x43, 0x6f, 0x6e, 0x73,
  0x74, 0x61, 0x6e, 0x74, 0x73, 0x00, 0x00, 0x00, 0x06, 0x00, 0x06, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6b, 0x65, 0x72, 0x6e,
  0x65, 0x6c, 0x53, 0x69, 0x7a, 0x65, 0x00, 0x00, 0x06, 0x00, 0x05, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x73, 0x69, 0x67, 0x6d,
  0x61, 0x00, 0x00, 0x00, 0x05, 0x00, 0x03, 0x00, 0x05, 0x00, 0x00, 0x00,
  0x70, 0x63, 0x00, 0x00, 0x05, 0x00, 0x06, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x74, 0x79, 0x70, 0x65, 0x2e, 0x32, 0x64, 0x2e, 0x69, 0x6d, 0x61, 0x67,
  0x65, 0x00, 0x00, 0x00, 0x05, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x73, 0x72, 0x63, 0x54, 0x65, 0x78, 0x00, 0x00, 0x05, 0x00, 0x06, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65, 0x2e, 0x73, 0x61, 0x6d,
  0x70, 0x6c, 0x65, 0x72, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x05, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x73, 0x72, 0x63, 0x53, 0x61, 0x6d, 0x70, 0x6c,
  0x65, 0x72, 0x00, 0x00, 0x05, 0x00, 0x06, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x74, 0x79, 0x70, 0x65, 0x2e, 0x32, 0x64, 0x2e, 0x69, 0x6d, 0x61, 0x67,
  0x65, 0x00, 0x00, 0x00, 0x05, 0x00, 0x05, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x64, 0x73, 0x74, 0x49, 0x6d, 0x61, 0x67, 0x65, 0x00, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x04, 0x00, 0x02, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e,
  0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x07, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x74, 0x79, 0x70, 0x65, 0x2e, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x65, 0x64,
  0x2e, 0x69, 0x6d, 0x61, 0x67, 0x65, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x05, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x2a, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x02, 0x00, 0x11, 0x00, 0x00, 0x00,
  0x29, 0x00, 0x03, 0x00, 0x11, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x16, 0x00, 0x03, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00,
  0x16, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x2c, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f, 0x17, 0x00, 0x04, 0x00,
  0x19, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x2c, 0x00, 0x05, 0x00, 0x19, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
  0x2b, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x80, 0x3f, 0x1e, 0x00, 0x04, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
  0x1d, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x19, 0x00, 0x09, 0x00, 0x06, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x04, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x02, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x04, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x19, 0x00, 0x09, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x17, 0x00, 0x04, 0x00, 0x22, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00,
  0x24, 0x00, 0x00, 0x00, 0x21, 0x00, 0x03, 0x00, 0x25, 0x00, 0x00, 0x00,
  0x24, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x26, 0x00, 0x00, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00,
  0x27, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x04, 0x00, 0x28, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x03, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x29, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
  0x1d, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x3b, 0x00, 0x04, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x1f, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x3b, 0x00, 0x04, 0x00, 0x23, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x21, 0x00, 0x00, 0x00,
  0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff,
  0x36, 0x00, 0x05, 0x00, 0x24, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x2c, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x22, 0x00, 0x00, 0x00,
  0x2d, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x03, 0x00,
  0x2e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfb, 0x00, 0x03, 0x00,
  0x2a, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x2f, 0x00, 0x00, 0x00, 0x4f, 0x00, 0x07, 0x00, 0x27, 0x00, 0x00, 0x00,
  0x30, 0x00, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00,
  0x26, 0x00, 0x00, 0x00, 0x31, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x68, 0x00, 0x04, 0x00, 0x27, 0x00, 0x00, 0x00,
  0x33, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x33, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x35, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00, 0x33, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x37, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00, 0x50, 0x00, 0x05, 0x00,
  0x26, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00,
  0x37, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x39, 0x00, 0x00, 0x00, 0x31, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xaf, 0x00, 0x05, 0x00, 0x11, 0x00, 0x00, 0x00, 0x3a, 0x00, 0x00, 0x00,
  0x39, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00, 0xa8, 0x00, 0x04, 0x00,
  0x11, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x00, 0x3a, 0x00, 0x00, 0x00,
  0xf7, 0x00, 0x03, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xfa, 0x00, 0x04, 0x00, 0x3b, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00,
  0x3c, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x3d, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x05, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00,
  0x31, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xaf, 0x00, 0x05, 0x00,
  0x11, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00,
  0x37, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00, 0x3c, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x3c, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00,
  0x11, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x2f, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00,
  0xf7, 0x00, 0x03, 0x00, 0x41, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xfa, 0x00, 0x04, 0x00, 0x40, 0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x00,
  0x41, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x42, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x2e, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x41, 0x00, 0x00, 0x00, 0x41, 0x00, 0x05, 0x00, 0x28, 0x00, 0x00, 0x00,
  0x43, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00,
  0x43, 0x00, 0x00, 0x00, 0x87, 0x00, 0x05, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x45, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x87, 0x00, 0x05, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x46, 0x00, 0x00, 0x00,
  0x44, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x47, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x47, 0x00, 0x00, 0x00,
  0xf5, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00,
  0x17, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00,
  0x4a, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x4b, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00,
  0x4c, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x4d, 0x00, 0x00, 0x00, 0x46, 0x00, 0x00, 0x00,
  0x41, 0x00, 0x00, 0x00, 0x4e, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00,
  0xb3, 0x00, 0x05, 0x00, 0x11, 0x00, 0x00, 0x00, 0x4f, 0x00, 0x00, 0x00,
  0x4d, 0x00, 0x00, 0x00, 0x45, 0x00, 0x00, 0x00, 0xf6, 0x00, 0x04, 0x00,
  0x50, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xfa, 0x00, 0x04, 0x00, 0x4f, 0x00, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00,
  0x50, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x51, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x52, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x52, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x4c, 0x00, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00,
  0x53, 0x00, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00,
  0x16, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00,
  0xf5, 0x00, 0x07, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x55, 0x00, 0x00, 0x00,
  0x46, 0x00, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00, 0x56, 0x00, 0x00, 0x00,
  0x54, 0x00, 0x00, 0x00, 0xb3, 0x00, 0x05, 0x00, 0x11, 0x00, 0x00, 0x00,
  0x57, 0x00, 0x00, 0x00, 0x55, 0x00, 0x00, 0x00, 0x45, 0x00, 0x00, 0x00,
  0xf6, 0x00, 0x04, 0x00, 0x58, 0x00, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00, 0x57, 0x00, 0x00, 0x00,
  0x54, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x54, 0x00, 0x00, 0x00, 0x50, 0x00, 0x05, 0x00, 0x26, 0x00, 0x00, 0x00,
  0x59, 0x00, 0x00, 0x00, 0x55, 0x00, 0x00, 0x00, 0x4d, 0x00, 0x00, 0x00,
  0x80, 0x00, 0x05, 0x00, 0x26, 0x00, 0x00, 0x00, 0x5a, 0x00, 0x00, 0x00,
  0x31, 0x00, 0x00, 0x00, 0x59, 0x00, 0x00, 0x00, 0x6f, 0x00, 0x04, 0x00,
  0x19, 0x00, 0x00, 0x00, 0x5b, 0x00, 0x00, 0x00, 0x5a, 0x00, 0x00, 0x00,
  0x81, 0x00, 0x05, 0x00, 0x19, 0x00, 0x00, 0x00, 0x5c, 0x00, 0x00, 0x00,
  0x5b, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x6f, 0x00, 0x04, 0x00,
  0x19, 0x00, 0x00, 0x00, 0x5d, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
  0x88, 0x00, 0x05, 0x00, 0x19, 0x00, 0x00, 0x00, 0x5e, 0x00, 0x00, 0x00,
  0x5c, 0x00, 0x00, 0x00, 0x5d, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x5f, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x56, 0x00, 0x05, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x61, 0x00, 0x00, 0x00, 0x5f, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00,
  0x58, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x62, 0x00, 0x00, 0x00,
  0x61, 0x00, 0x00, 0x00, 0x5e, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x00, 0x00, 0x84, 0x00, 0x05, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x63, 0x00, 0x00, 0x00, 0x55, 0x00, 0x00, 0x00, 0x55, 0x00, 0x00, 0x00,
  0x84, 0x00, 0x05, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
  0x4d, 0x00, 0x00, 0x00, 0x4d, 0x00, 0x00, 0x00, 0x80, 0x00, 0x05, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x65, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00,
  0x64, 0x00, 0x00, 0x00, 0x6f, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x66, 0x00, 0x00, 0x00, 0x65, 0x00, 0x00, 0x00, 0x7f, 0x00, 0x04, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00, 0x66, 0x00, 0x00, 0x00,
  0x41, 0x00, 0x05, 0x00, 0x29, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x69, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
  0x85, 0x00, 0x05, 0x00, 0x14, 0x00, 0x00, 0x00, 0x6a, 0x00, 0x00, 0x00,
  0x69, 0x00, 0x00, 0x00, 0x69, 0x00, 0x00, 0x00, 0x85, 0x00, 0x05, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00, 0x6a, 0x00, 0x00, 0x00,
  0x1b, 0x00, 0x00, 0x00, 0x88, 0x00, 0x05, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x6c, 0x00, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x06, 0x00, 0x14, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00,
  0x50, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x6e, 0x00, 0x00, 0x00,
  0x6d, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00,
  0x6d, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00,
  0x62, 0x00, 0x00, 0x00, 0x6e, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00,
  0x81, 0x00, 0x05, 0x00, 0x14, 0x00, 0x00, 0x00, 0x53, 0x00, 0x00, 0x00,
  0x4c, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00, 0x80, 0x00, 0x05, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x56, 0x00, 0x00, 0x00, 0x55, 0x00, 0x00, 0x00,
  0x10, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00, 0x52, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x58, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x4a, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x4a, 0x00, 0x00, 0x00,
  0x80, 0x00, 0x05, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x4e, 0x00, 0x00, 0x00,
  0x4d, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x47, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x50, 0x00, 0x00, 0x00,
  0x50, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x6f, 0x00, 0x00, 0x00,
  0x4b, 0x00, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x00,
  0x4b, 0x00, 0x00, 0x00, 0x88, 0x00, 0x05, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x70, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00, 0x6f, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x05, 0x00, 0x14, 0x00, 0x00, 0x00, 0x71, 0x00, 0x00, 0x00,
  0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x72, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x73, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x50, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x74, 0x00, 0x00, 0x00,
  0x71, 0x00, 0x00, 0x00, 0x72, 0x00, 0x00, 0x00, 0x73, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00, 0x27, 0x00, 0x00, 0x00,
  0x75, 0x00, 0x00, 0x00, 0x31, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x76, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x63, 0x00, 0x05, 0x00, 0x76, 0x00, 0x00, 0x00, 0x75, 0x00, 0x00, 0x00,
  0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x2e, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x2e, 0x00, 0x00, 0x00,
  0xfd, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00
};
