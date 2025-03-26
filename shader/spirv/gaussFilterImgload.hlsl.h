#if 0
; SPIR-V
; Version: 1.0
; Generator: Google spiregg; 0
; Bound: 111
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
               OpName %pc "pc"
               OpName %type_2d_image "type.2d.image"
               OpName %srcTex "srcTex"
               OpName %type_2d_image_0 "type.2d.image"
               OpName %dstImage "dstImage"
               OpName %type_ConstantBuffer_UBO "type.ConstantBuffer.UBO"
               OpMemberName %type_ConstantBuffer_UBO 0 "weights"
               OpName %ubo "ubo"
               OpName %main "main"
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %srcTex DescriptorSet 0
               OpDecorate %srcTex Binding 0
               OpDecorate %dstImage DescriptorSet 0
               OpDecorate %dstImage Binding 1
               OpDecorate %ubo DescriptorSet 0
               OpDecorate %ubo Binding 2
               OpMemberDecorate %type_ConstantBuffer_PushConstants 0 Offset 0
               OpDecorate %type_ConstantBuffer_PushConstants Block
               OpDecorate %_arr_v4float_uint_4 ArrayStride 16
               OpMemberDecorate %type_ConstantBuffer_UBO 0 Offset 0
               OpDecorate %type_ConstantBuffer_UBO Block
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
       %bool = OpTypeBool
       %true = OpConstantTrue %bool
      %float = OpTypeFloat 32
    %float_1 = OpConstant %float 1
      %int_2 = OpConstant %int 2
    %float_0 = OpConstant %float 0
    %v3float = OpTypeVector %float 3
         %23 = OpConstantComposite %v3float %float_0 %float_0 %float_0
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
      %int_3 = OpConstant %int 3
%type_ConstantBuffer_PushConstants = OpTypeStruct %int
%_ptr_PushConstant_type_ConstantBuffer_PushConstants = OpTypePointer PushConstant %type_ConstantBuffer_PushConstants
%type_2d_image = OpTypeImage %float 2D 2 0 0 1 Unknown
%_ptr_UniformConstant_type_2d_image = OpTypePointer UniformConstant %type_2d_image
%type_2d_image_0 = OpTypeImage %float 2D 2 0 0 2 Rgba8
%_ptr_UniformConstant_type_2d_image_0 = OpTypePointer UniformConstant %type_2d_image_0
     %uint_4 = OpConstant %uint 4
    %v4float = OpTypeVector %float 4
%_arr_v4float_uint_4 = OpTypeArray %v4float %uint_4
%type_ConstantBuffer_UBO = OpTypeStruct %_arr_v4float_uint_4
%_ptr_Uniform_type_ConstantBuffer_UBO = OpTypePointer Uniform %type_ConstantBuffer_UBO
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %36 = OpTypeFunction %void
      %v2int = OpTypeVector %int 2
     %v2uint = OpTypeVector %uint 2
%_ptr_PushConstant_int = OpTypePointer PushConstant %int
%_ptr_Uniform_float = OpTypePointer Uniform %float
         %pc = OpVariable %_ptr_PushConstant_type_ConstantBuffer_PushConstants PushConstant
     %srcTex = OpVariable %_ptr_UniformConstant_type_2d_image UniformConstant
   %dstImage = OpVariable %_ptr_UniformConstant_type_2d_image_0 UniformConstant
        %ubo = OpVariable %_ptr_Uniform_type_ConstantBuffer_UBO Uniform
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
     %int_n2 = OpConstant %int -2
       %main = OpFunction %void None %36
         %42 = OpLabel
         %43 = OpLoad %v3uint %gl_GlobalInvocationID
               OpSelectionMerge %44 None
               OpSwitch %uint_0 %45
         %45 = OpLabel
         %46 = OpVectorShuffle %v2uint %43 %43 0 1
         %47 = OpBitcast %v2int %46
         %48 = OpLoad %type_2d_image_0 %dstImage
         %49 = OpImageQuerySize %v2uint %48
         %50 = OpCompositeExtract %uint %49 0
         %51 = OpBitcast %int %50
         %52 = OpCompositeExtract %uint %49 1
         %53 = OpBitcast %int %52
         %54 = OpCompositeExtract %int %47 0
         %55 = OpSGreaterThanEqual %bool %54 %51
         %56 = OpLogicalNot %bool %55
               OpSelectionMerge %57 None
               OpBranchConditional %56 %58 %57
         %58 = OpLabel
         %59 = OpCompositeExtract %int %47 1
         %60 = OpSGreaterThanEqual %bool %59 %53
               OpBranch %57
         %57 = OpLabel
         %61 = OpPhi %bool %true %45 %60 %58
               OpSelectionMerge %62 None
               OpBranchConditional %61 %63 %62
         %63 = OpLabel
               OpBranch %44
         %62 = OpLabel
         %64 = OpLoad %type_2d_image %srcTex
         %65 = OpAccessChain %_ptr_PushConstant_int %pc %int_0
         %66 = OpLoad %int %65
         %67 = OpSDiv %int %66 %int_2
         %68 = OpSDiv %int %66 %int_n2
               OpBranch %69
         %69 = OpLabel
         %70 = OpPhi %v3float %23 %62 %71 %72
         %73 = OpPhi %int %68 %62 %74 %72
         %75 = OpSLessThanEqual %bool %73 %67
               OpLoopMerge %76 %72 None
               OpBranchConditional %75 %77 %76
         %77 = OpLabel
               OpBranch %78
         %78 = OpLabel
         %79 = OpPhi %v3float %23 %77 %80 %81
         %82 = OpPhi %int %68 %77 %83 %81
         %84 = OpSLessThanEqual %bool %82 %67
               OpLoopMerge %85 %81 None
               OpBranchConditional %84 %81 %85
         %81 = OpLabel
         %86 = OpCompositeConstruct %v2int %82 %73
         %87 = OpIAdd %v2int %47 %86
         %88 = OpBitcast %v2uint %87
         %89 = OpImageFetch %v4float %64 %88 Lod %uint_0
         %90 = OpExtInst %int %1 SAbs %82
         %91 = OpShiftRightArithmetic %int %90 %int_2
         %92 = OpBitwiseAnd %int %90 %int_3
         %93 = OpBitcast %uint %92
         %94 = OpAccessChain %_ptr_Uniform_float %ubo %int_0 %91 %93
         %95 = OpLoad %float %94
         %96 = OpVectorShuffle %v3float %89 %89 0 1 2
         %97 = OpVectorTimesScalar %v3float %96 %95
         %80 = OpFAdd %v3float %79 %97
         %83 = OpIAdd %int %82 %int_1
               OpBranch %78
         %85 = OpLabel
         %98 = OpExtInst %int %1 SAbs %73
         %99 = OpShiftRightArithmetic %int %98 %int_2
        %100 = OpBitwiseAnd %int %98 %int_3
        %101 = OpBitcast %uint %100
        %102 = OpAccessChain %_ptr_Uniform_float %ubo %int_0 %99 %101
        %103 = OpLoad %float %102
        %104 = OpVectorTimesScalar %v3float %79 %103
         %71 = OpFAdd %v3float %70 %104
               OpBranch %72
         %72 = OpLabel
         %74 = OpIAdd %int %73 %int_1
               OpBranch %69
         %76 = OpLabel
        %105 = OpCompositeExtract %float %70 0
        %106 = OpCompositeExtract %float %70 1
        %107 = OpCompositeExtract %float %70 2
        %108 = OpCompositeConstruct %v4float %105 %106 %107 %float_1
        %109 = OpBitcast %v2uint %47
        %110 = OpLoad %type_2d_image_0 %dstImage
               OpImageWrite %110 %109 %108 None
               OpBranch %44
         %44 = OpLabel
               OpReturn
               OpFunctionEnd

#endif

const unsigned char g_main[] = {
  0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00,
  0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00,
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
  0x65, 0x6c, 0x53, 0x69, 0x7a, 0x65, 0x00, 0x00, 0x05, 0x00, 0x03, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x70, 0x63, 0x00, 0x00, 0x05, 0x00, 0x06, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65, 0x2e, 0x32, 0x64, 0x2e,
  0x69, 0x6d, 0x61, 0x67, 0x65, 0x00, 0x00, 0x00, 0x05, 0x00, 0x04, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x73, 0x72, 0x63, 0x54, 0x65, 0x78, 0x00, 0x00,
  0x05, 0x00, 0x06, 0x00, 0x08, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65,
  0x2e, 0x32, 0x64, 0x2e, 0x69, 0x6d, 0x61, 0x67, 0x65, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x05, 0x00, 0x09, 0x00, 0x00, 0x00, 0x64, 0x73, 0x74, 0x49,
  0x6d, 0x61, 0x67, 0x65, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x08, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x74, 0x79, 0x70, 0x65, 0x2e, 0x43, 0x6f, 0x6e,
  0x73, 0x74, 0x61, 0x6e, 0x74, 0x42, 0x75, 0x66, 0x66, 0x65, 0x72, 0x2e,
  0x55, 0x42, 0x4f, 0x00, 0x06, 0x00, 0x05, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x77, 0x65, 0x69, 0x67, 0x68, 0x74, 0x73, 0x00,
  0x05, 0x00, 0x03, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x75, 0x62, 0x6f, 0x00,
  0x05, 0x00, 0x04, 0x00, 0x02, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x47, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
  0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x02, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x29, 0x00, 0x03, 0x00, 0x10, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
  0x16, 0x00, 0x03, 0x00, 0x12, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x80, 0x3f, 0x2b, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x17, 0x00, 0x04, 0x00, 0x16, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x06, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x17, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00,
  0x15, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
  0x03, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x03, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x1b, 0x00, 0x00, 0x00,
  0x09, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x19, 0x00, 0x09, 0x00,
  0x06, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
  0x19, 0x00, 0x09, 0x00, 0x08, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x04, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x1e, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00,
  0x1f, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
  0x1c, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00,
  0x1e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x03, 0x00, 0x0a, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x04, 0x00, 0x22, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x21, 0x00, 0x03, 0x00, 0x24, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x17, 0x00, 0x04, 0x00, 0x25, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x26, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
  0x27, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x20, 0x00, 0x04, 0x00, 0x28, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x1b, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
  0x1c, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x3b, 0x00, 0x04, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x20, 0x00, 0x00, 0x00,
  0x0b, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
  0x22, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x29, 0x00, 0x00, 0x00,
  0xfe, 0xff, 0xff, 0xff, 0x36, 0x00, 0x05, 0x00, 0x23, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x21, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0xf7, 0x00, 0x03, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0xfb, 0x00, 0x03, 0x00, 0x19, 0x00, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x2d, 0x00, 0x00, 0x00, 0x4f, 0x00, 0x07, 0x00,
  0x26, 0x00, 0x00, 0x00, 0x2e, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x00, 0x00,
  0x2b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x25, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
  0x2e, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00,
  0x30, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x68, 0x00, 0x04, 0x00,
  0x26, 0x00, 0x00, 0x00, 0x31, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x05, 0x00, 0x18, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00,
  0x31, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x33, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x05, 0x00, 0x18, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
  0x31, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x05, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00,
  0x2f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xaf, 0x00, 0x05, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x37, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00,
  0x33, 0x00, 0x00, 0x00, 0xa8, 0x00, 0x04, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x38, 0x00, 0x00, 0x00, 0x37, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x03, 0x00,
  0x39, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00,
  0x38, 0x00, 0x00, 0x00, 0x3a, 0x00, 0x00, 0x00, 0x39, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x3a, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0xaf, 0x00, 0x05, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x3c, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00,
  0xf9, 0x00, 0x02, 0x00, 0x39, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
  0x39, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x00,
  0x3c, 0x00, 0x00, 0x00, 0x3a, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x03, 0x00,
  0x3e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00,
  0x3d, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x3f, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x2c, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x3e, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
  0x07, 0x00, 0x00, 0x00, 0x41, 0x00, 0x05, 0x00, 0x27, 0x00, 0x00, 0x00,
  0x41, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x00,
  0x41, 0x00, 0x00, 0x00, 0x87, 0x00, 0x05, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x43, 0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00,
  0x87, 0x00, 0x05, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00,
  0x42, 0x00, 0x00, 0x00, 0x29, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x45, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x45, 0x00, 0x00, 0x00,
  0xf5, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x46, 0x00, 0x00, 0x00,
  0x17, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00, 0x47, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x49, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00,
  0x4a, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00, 0xb3, 0x00, 0x05, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00,
  0x43, 0x00, 0x00, 0x00, 0xf6, 0x00, 0x04, 0x00, 0x4c, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00,
  0x4b, 0x00, 0x00, 0x00, 0x4d, 0x00, 0x00, 0x00, 0x4c, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x4d, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x4e, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x4e, 0x00, 0x00, 0x00,
  0xf5, 0x00, 0x07, 0x00, 0x16, 0x00, 0x00, 0x00, 0x4f, 0x00, 0x00, 0x00,
  0x17, 0x00, 0x00, 0x00, 0x4d, 0x00, 0x00, 0x00, 0x50, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x00, 0x00, 0xf5, 0x00, 0x07, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x52, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x4d, 0x00, 0x00, 0x00,
  0x53, 0x00, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00, 0xb3, 0x00, 0x05, 0x00,
  0x10, 0x00, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00, 0x52, 0x00, 0x00, 0x00,
  0x43, 0x00, 0x00, 0x00, 0xf6, 0x00, 0x04, 0x00, 0x55, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xfa, 0x00, 0x04, 0x00,
  0x54, 0x00, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00, 0x55, 0x00, 0x00, 0x00,
  0xf8, 0x00, 0x02, 0x00, 0x51, 0x00, 0x00, 0x00, 0x50, 0x00, 0x05, 0x00,
  0x25, 0x00, 0x00, 0x00, 0x56, 0x00, 0x00, 0x00, 0x52, 0x00, 0x00, 0x00,
  0x49, 0x00, 0x00, 0x00, 0x80, 0x00, 0x05, 0x00, 0x25, 0x00, 0x00, 0x00,
  0x57, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x56, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x26, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00,
  0x57, 0x00, 0x00, 0x00, 0x5f, 0x00, 0x07, 0x00, 0x1f, 0x00, 0x00, 0x00,
  0x59, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00,
  0x02, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x06, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x5a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x05, 0x00, 0x00, 0x00, 0x52, 0x00, 0x00, 0x00, 0xc3, 0x00, 0x05, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x5b, 0x00, 0x00, 0x00, 0x5a, 0x00, 0x00, 0x00,
  0x14, 0x00, 0x00, 0x00, 0xc7, 0x00, 0x05, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x5c, 0x00, 0x00, 0x00, 0x5a, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
  0x7c, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00, 0x5d, 0x00, 0x00, 0x00,
  0x5c, 0x00, 0x00, 0x00, 0x41, 0x00, 0x07, 0x00, 0x28, 0x00, 0x00, 0x00,
  0x5e, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
  0x5b, 0x00, 0x00, 0x00, 0x5d, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x5f, 0x00, 0x00, 0x00, 0x5e, 0x00, 0x00, 0x00,
  0x4f, 0x00, 0x08, 0x00, 0x16, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00,
  0x59, 0x00, 0x00, 0x00, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x8e, 0x00, 0x05, 0x00,
  0x16, 0x00, 0x00, 0x00, 0x61, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00,
  0x5f, 0x00, 0x00, 0x00, 0x81, 0x00, 0x05, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x50, 0x00, 0x00, 0x00, 0x4f, 0x00, 0x00, 0x00, 0x61, 0x00, 0x00, 0x00,
  0x80, 0x00, 0x05, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x53, 0x00, 0x00, 0x00,
  0x52, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x4e, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x55, 0x00, 0x00, 0x00,
  0x0c, 0x00, 0x06, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x62, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x49, 0x00, 0x00, 0x00,
  0xc3, 0x00, 0x05, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00,
  0x62, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0xc7, 0x00, 0x05, 0x00,
  0x0d, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x62, 0x00, 0x00, 0x00,
  0x1a, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00,
  0x65, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x41, 0x00, 0x07, 0x00,
  0x28, 0x00, 0x00, 0x00, 0x66, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
  0x0e, 0x00, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00, 0x65, 0x00, 0x00, 0x00,
  0x3d, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00,
  0x66, 0x00, 0x00, 0x00, 0x8e, 0x00, 0x05, 0x00, 0x16, 0x00, 0x00, 0x00,
  0x68, 0x00, 0x00, 0x00, 0x4f, 0x00, 0x00, 0x00, 0x67, 0x00, 0x00, 0x00,
  0x81, 0x00, 0x05, 0x00, 0x16, 0x00, 0x00, 0x00, 0x47, 0x00, 0x00, 0x00,
  0x46, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x48, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x48, 0x00, 0x00, 0x00,
  0x80, 0x00, 0x05, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00,
  0x49, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x45, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x4c, 0x00, 0x00, 0x00,
  0x51, 0x00, 0x05, 0x00, 0x12, 0x00, 0x00, 0x00, 0x69, 0x00, 0x00, 0x00,
  0x46, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
  0x12, 0x00, 0x00, 0x00, 0x6a, 0x00, 0x00, 0x00, 0x46, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x12, 0x00, 0x00, 0x00,
  0x6b, 0x00, 0x00, 0x00, 0x46, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x50, 0x00, 0x07, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00,
  0x69, 0x00, 0x00, 0x00, 0x6a, 0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x04, 0x00, 0x26, 0x00, 0x00, 0x00,
  0x6d, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x6e, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
  0x63, 0x00, 0x05, 0x00, 0x6e, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00,
  0x6c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf9, 0x00, 0x02, 0x00,
  0x2c, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x2c, 0x00, 0x00, 0x00,
  0xfd, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00
};
