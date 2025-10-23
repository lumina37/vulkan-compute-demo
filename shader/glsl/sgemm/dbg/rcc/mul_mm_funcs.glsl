// mul_mm_funcs.glsl

void load_a_to_shmem(const uint pos_a, const uint row, const uint col, const uint idx_m, const uint block, const uint end_k) {
#if defined(DATA_A_F32) || defined(DATA_A_F16)
#if LOAD_VEC_A == 8
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;
            FLOAT_TYPE_VEC8 aa = FLOAT_TYPE_VEC8(data_a[idx]);
            buf_a[buf_idx    ] = aa[0].xy;
            buf_a[buf_idx + 1] = aa[0].zw;
            buf_a[buf_idx + 2] = aa[1].xy;
            buf_a[buf_idx + 3] = aa[1].zw;
#elif LOAD_VEC_A == 4
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;
            FLOAT_TYPE_VEC4 aa = FLOAT_TYPE_VEC4(data_a[idx]);
            buf_a[buf_idx    ] = aa.xy;
            buf_a[buf_idx + 1] = aa.zw;
#else // LOAD_VEC_BATCH_A == 2
            const uint idx = pos_a + col * p.stride_a + row * 2;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (idx_m < p.M && block + row * 2 + 1 < end_k) {
                buf_a[buf_idx] = FLOAT_TYPE_VEC2(data_a[idx],
                                                 data_a[idx + 1]);
            } else if (idx_m < p.M && block + row * 2 < end_k) {
                buf_a[buf_idx] = FLOAT_TYPE_VEC2(data_a[idx], 0.0f);
            } else {
                buf_a[buf_idx] = FLOAT_TYPE_VEC2(0.0f);
            }
#endif
#endif
}

void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint idx_n, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
            // Not supported for b_type bf16 because bf16mat2x4 does not exist
            const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
            FLOAT_TYPE_VEC8 bb = FLOAT_TYPE_VEC8(data_b[idx]);
            buf_b[buf_idx + 0] = bb[0].xy;
            buf_b[buf_idx + 1] = bb[0].zw;
            buf_b[buf_idx + 2] = bb[1].xy;
            buf_b[buf_idx + 3] = bb[1].zw;
#elif LOAD_VEC_B == 4
            const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
            FLOAT_TYPE_VEC4 bb = FLOAT_TYPE_VEC4(data_b[idx]);
            buf_b[buf_idx + 0] = bb.xy;
            buf_b[buf_idx + 1] = bb.zw;
#else // LOAD_VEC_BATCH_B == 2
            const uint idx = pos_b + col * p.stride_b + row * 2;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (idx_n < p.N && block + row * 2 + 1 < end_k) {
                buf_b[buf_idx] = FLOAT_TYPE_VEC2(TO_FLOAT_TYPE(data_b[idx]),
                                                 TO_FLOAT_TYPE(data_b[idx + 1]));
            } else if (idx_n < p.N && block + row * 2 < end_k) {
                buf_b[buf_idx] = FLOAT_TYPE_VEC2(TO_FLOAT_TYPE(data_b[idx]), 0.0f);
            } else {
                buf_b[buf_idx] = FLOAT_TYPE_VEC2(0.0f);
            }
#endif
}