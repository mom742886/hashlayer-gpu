#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

#define BLAKE2B_HASH_LENGTH 64
#define BLAKE2B_BLOCK_SIZE 128
#define BLAKE2B_QWORDS_IN_BLOCK (BLAKE2B_BLOCK_SIZE / 8)

#define MAX_BLOCK_HEADER_SIZE 128

#ifdef _WIN32
#pragma pack(push, 1)
#endif

struct
#ifndef _WIN32
    __attribute__((packed))
#endif
    block_header
{
    uint8_t data[MAX_BLOCK_HEADER_SIZE];
    uint32_t data_len;
};

#ifdef _WIN32
#pragma pack(pop)
#endif

// TODO Rename/split/move to another class
struct worker_t
{
    uint32_t nonces_per_run;
    uint64_t **block_header_data;
    uint32_t **nonce;
    uint32_t header_len;
    uint32_t nonce_offset;
    uint32_t difficulty_bits;
    dim3 mine_blocks;
    dim3 mine_threads;
};

__global__ void mine_blake2b(uint64_t *block_header_data, uint32_t block_header_len, uint32_t nonce_offset, uint32_t start_nonce, uint32_t difficulty_bits, uint32_t *nonce);

__host__ void set_block_header(struct worker_t *worker, uint32_t threadIndex, block_header *header);
__host__ cudaError_t mine_nonces(struct worker_t *worker, uint32_t threadIndex, uint32_t start_nonce, uint32_t *nonce);

#endif