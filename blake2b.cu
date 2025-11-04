#include "kernels.h"

#define IV0 0x6a09e667f3bcc908UL
#define IV1 0xbb67ae8584caa73bUL
#define IV2 0x3c6ef372fe94f82bUL
#define IV3 0xa54ff53a5f1d36f1UL
#define IV4 0x510e527fade682d1UL
#define IV5 0x9b05688c2b3e6c1fUL
#define IV6 0x1f83d9abfb41bd6bUL
#define IV7 0x5be0cd19137e2179UL

__constant__ static const uint8_t sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

__device__ __forceinline__ uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64 - n));
}

__device__ __forceinline__ uint64_t swap64(uint64_t x)
{
    return ((x & 0x00000000000000FFUL) << 56)
        | ((x & 0x000000000000FF00UL) << 40)
        | ((x & 0x0000000000FF0000UL) << 24)
        | ((x & 0x00000000FF000000UL) <<  8)
        | ((x & 0x000000FF00000000UL) >>  8)
        | ((x & 0x0000FF0000000000UL) >> 24)
        | ((x & 0x00FF000000000000UL) >> 40)
        | ((x & 0xFF00000000000000UL) >> 56);
}

__device__ void blake2b_init(uint64_t *h, uint32_t hashlen)
{
    h[0] = IV0 ^ (0x01010000 | hashlen);
    h[1] = IV1;
    h[2] = IV2;
    h[3] = IV3;
    h[4] = IV4;
    h[5] = IV5;
    h[6] = IV6;
    h[7] = IV7;
}

__device__ void g(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, uint64_t m1, uint64_t m2)
{
    asm("{"
        ".reg .u64 s, x;"
        ".reg .u32 l1, l2, h1, h2;"
        // a = a + b + x
        "add.u64 %0, %0, %1;"
        "add.u64 %0, %0, %4;"
        // d = rotr64(d ^ a, 32)
        "xor.b64 x, %3, %0;"
        "mov.b64 {h1, l1}, x;"
        "mov.b64 %3, {l1, h1};"
        // c = c + d
        "add.u64 %2, %2, %3;"
        // b = rotr64(b ^ c, 24)
        "xor.b64 x, %1, %2;"
        "mov.b64 {l1, h1}, x;"
        "prmt.b32 l2, l1, h1, 0x6543;"
        "prmt.b32 h2, l1, h1, 0x2107;"
        "mov.b64 %1, {l2, h2};"
        // a = a + b + y
        "add.u64 %0, %0, %1;"
        "add.u64 %0, %0, %5;"
        // d = rotr64(d ^ a, 16);
        "xor.b64 x, %3, %0;"
        "mov.b64 {l1, h1}, x;"
        "prmt.b32 l2, l1, h1, 0x5432;"
        "prmt.b32 h2, l1, h1, 0x1076;"
        "mov.b64 %3, {l2, h2};"
        // c = c + d
        "add.u64 %2, %2, %3;"
        // b = rotr64(b ^ c, 63)
        "xor.b64 x, %1, %2;"
        "shl.b64 s, x, 1;"
        "shr.b64 x, x, 63;"
        "add.u64 %1, s, x;"
        "}"
        : "+l"(*a), "+l"(*b), "+l"(*c), "+l"(*d) : "l"(m1), "l"(m2)
    );
}

#define G(i, a, b, c, d) (g(&v[a], &v[b], &v[c], &v[d], m[sigma[r][2 * i]], m[sigma[r][2 * i + 1]]))

__device__ void blake2b_round(uint32_t r, uint64_t *v, uint64_t *m)
{
    G(0, 0, 4, 8, 12);
    G(1, 1, 5, 9, 13);
    G(2, 2, 6, 10, 14);
    G(3, 3, 7, 11, 15);
    G(4, 0, 5, 10, 15);
    G(5, 1, 6, 11, 12);
    G(6, 2, 7, 8, 13);
    G(7, 3, 4, 9, 14);
}

__device__ void blake2b_compress(uint64_t *h, uint64_t *m, uint32_t bytes_compressed, bool last_block)
{
    uint64_t v[BLAKE2B_QWORDS_IN_BLOCK];

    v[0] = h[0];
    v[1] = h[1];
    v[2] = h[2];
    v[3] = h[3];
    v[4] = h[4];
    v[5] = h[5];
    v[6] = h[6];
    v[7] = h[7];
    v[8] = IV0;
    v[9] = IV1;
    v[10] = IV2;
    v[11] = IV3;
    v[12] = IV4 ^ bytes_compressed;
    v[13] = IV5; // it's OK if below 2^32 bytes
    v[14] = last_block ? ~IV6 : IV6;
    v[15] = IV7;

    #pragma unroll
    for (uint32_t r = 0; r < 12; r++)
    {
        blake2b_round(r, v, m);
    }

    h[0] = h[0] ^ v[0] ^ v[8];
    h[1] = h[1] ^ v[1] ^ v[9];
    h[2] = h[2] ^ v[2] ^ v[10];
    h[3] = h[3] ^ v[3] ^ v[11];
    h[4] = h[4] ^ v[4] ^ v[12];
    h[5] = h[5] ^ v[5] ^ v[13];
    h[6] = h[6] ^ v[6] ^ v[14];
    h[7] = h[7] ^ v[7] ^ v[15];
}

__device__ void blake2b_hash(uint8_t *input, uint32_t input_len, uint64_t *output)
{
    uint64_t h[8];
    uint64_t buffer[BLAKE2B_QWORDS_IN_BLOCK];
    
    blake2b_init(h, BLAKE2B_HASH_LENGTH);
    
    uint32_t bytes_processed = 0;
    
    // Process full blocks
    while (bytes_processed + BLAKE2B_BLOCK_SIZE <= input_len)
    {
        // Convert input bytes to uint64_t words (little-endian)
        for (uint32_t i = 0; i < BLAKE2B_QWORDS_IN_BLOCK; i++)
        {
            buffer[i] = 0;
            uint32_t byte_offset = bytes_processed + i * 8;
            for (uint32_t j = 0; j < 8 && (byte_offset + j) < input_len; j++)
            {
                buffer[i] |= ((uint64_t)input[byte_offset + j]) << (j * 8);
            }
        }
        
        blake2b_compress(h, buffer, bytes_processed, false);
        bytes_processed += BLAKE2B_BLOCK_SIZE;
    }
    
    // Process remaining bytes
    uint32_t remaining = input_len - bytes_processed;
    if (remaining > 0 || bytes_processed == 0)
    {
        for (uint32_t i = 0; i < BLAKE2B_QWORDS_IN_BLOCK; i++)
        {
            buffer[i] = 0;
            uint32_t byte_offset = bytes_processed + i * 8;
            for (uint32_t j = 0; j < 8 && (byte_offset + j) < input_len; j++)
            {
                buffer[i] |= ((uint64_t)input[byte_offset + j]) << (j * 8);
            }
        }
        blake2b_compress(h, buffer, input_len, true);
    }
    else
    {
        // Finalize with empty last block
        for (uint32_t i = 0; i < BLAKE2B_QWORDS_IN_BLOCK; i++)
        {
            buffer[i] = 0;
        }
        blake2b_compress(h, buffer, bytes_processed, true);
    }
    
    // Copy output
    for (uint32_t i = 0; i < 8; i++)
    {
        output[i] = h[i];
    }
}

__device__ __forceinline__ bool has_leading_zero_bits(uint8_t *hash_bytes, uint32_t difficulty_bits)
{
    // Use bit shifting instead of division/modulo (8 = 2^3)
    uint32_t full_zero_bytes = difficulty_bits >> 3;
    uint32_t partial_bits = difficulty_bits & 7;
    
    // Early exit if difficulty exceeds hash length
    if (full_zero_bytes >= 64) return false;

    // Optimize: check uint64_t at a time for full bytes (8 bytes per check)
    // This reduces the number of comparisons significantly
    uint64_t *hash_qwords = (uint64_t *)hash_bytes;
    uint32_t full_zero_qwords = full_zero_bytes >> 3;
    uint32_t remaining_bytes = full_zero_bytes & 7;
    
    // Check full uint64_t words (8 bytes each)
    #pragma unroll
    for (uint32_t i = 0; i < full_zero_qwords; i++)
    {
        if (hash_qwords[i] != 0) return false;
    }
    
    // Check remaining full bytes (using byte comparison for remaining < 8 bytes)
    uint32_t byte_offset = full_zero_qwords << 3;
    #pragma unroll
    for (uint32_t i = 0; i < remaining_bytes; i++)
    {
        if (hash_bytes[byte_offset + i] != 0) return false;
    }

    // Check partial bits if needed
    if (partial_bits > 0)
    {
        uint32_t check_byte_idx = full_zero_bytes;
        if (check_byte_idx >= 64) return false;
        
        uint8_t byte = hash_bytes[check_byte_idx];
        // Optimize: mask calculation at compile time where possible
        // mask = (0xFF << (8 - partial_bits)) & 0xFF
        // For partial_bits = 1: mask = 0x80 (10000000)
        // For partial_bits = 2: mask = 0xC0 (11000000)
        // etc.
        uint8_t mask = (uint8_t)(0xFF << (8 - partial_bits));
        if ((byte & mask) != 0) return false;
    }

    return true;
}

__device__ __forceinline__ void set_nonce_in_header(uint8_t *header, uint32_t header_len, uint32_t nonce_offset, uint32_t nonce)
{
    // Set nonce at the specified offset (assuming nonce_offset is byte offset)
    if (nonce_offset + 4 <= header_len)
    {
        *((uint32_t*)(header + nonce_offset)) = nonce;
    }
}

__global__ void mine_blake2b(uint64_t *block_header_data, uint32_t block_header_len, uint32_t nonce_offset, uint32_t start_nonce, uint32_t difficulty_bits, uint32_t *nonce)
{
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t current_nonce = start_nonce + thread_id;
    
    // Early exit if solution already found
    if (atomicAdd(nonce, 0) != 0)
    {
        return;
    }
    
    // Convert block_header_data to bytes for hashing
    uint8_t header_bytes[MAX_BLOCK_HEADER_SIZE];
    
    // Copy header data
    for (uint32_t i = 0; i < block_header_len; i++)
    {
        uint32_t qword_idx = i / 8;
        uint32_t byte_idx = i % 8;
        header_bytes[i] = (uint8_t)((block_header_data[qword_idx] >> (byte_idx * 8)) & 0xFF);
    }
    
    // Set nonce at the specified offset
    set_nonce_in_header(header_bytes, block_header_len, nonce_offset, current_nonce);
    
    // Hash the block header
    uint64_t hash[8];
    blake2b_hash(header_bytes, block_header_len, hash);
    
    // Convert hash from uint64_t[8] to uint8_t[64] for bit checking
    uint8_t hash_bytes[64];
    for (uint32_t i = 0; i < 8; i++)
    {
        uint64_t value = hash[i];
        for (uint32_t j = 0; j < 8; j++)
        {
            hash_bytes[i * 8 + j] = (uint8_t)(value & 0xFF);
            value >>= 8;
        }
    }
    
    // Check proof of work: verify leading zero bits
    if (has_leading_zero_bits(hash_bytes, difficulty_bits))
    {
        atomicCAS(nonce, 0, current_nonce);
    }
}