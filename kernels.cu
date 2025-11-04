#include "kernels.h"
#include <string.h>
#include <stdlib.h>

__host__ void set_block_header(struct worker_t *worker, uint32_t threadIndex, block_header *header)
{
    // Store header length in worker
    worker->header_len = header->data_len;
    
    // Convert block header bytes to uint64_t array for GPU
    uint32_t qwords_needed = (header->data_len + 7) / 8;
    uint64_t *header_qwords = (uint64_t*)malloc(qwords_needed * sizeof(uint64_t));
    
    // Clear and copy data
    memset(header_qwords, 0, qwords_needed * sizeof(uint64_t));
    memcpy(header_qwords, header->data, header->data_len);
    
    cudaMemcpyAsync(worker->block_header_data[threadIndex], header_qwords, 
                    qwords_needed * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemsetAsync(worker->nonce[threadIndex], 0, sizeof(uint32_t)); // zero nonce
    
    free(header_qwords);
}

__host__ cudaError_t mine_nonces(struct worker_t *worker, uint32_t threadIndex, uint32_t start_nonce, uint32_t *nonce)
{
    mine_blake2b<<<worker->mine_blocks, worker->mine_threads>>>(
        worker->block_header_data[threadIndex], 
        worker->header_len,
        worker->nonce_offset,
        start_nonce, 
        worker->difficulty_bits,
        worker->nonce[threadIndex]);

    cudaError_t result = cudaStreamSynchronize(0);
    if (result != cudaSuccess)
    {
        return result;
    }

    cudaMemcpy(nonce, worker->nonce[threadIndex], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (*nonce > 0)
    {
        cudaMemsetAsync(worker->nonce[threadIndex], 0, sizeof(uint32_t)); // zero nonce
    }

    return cudaSuccess;
}