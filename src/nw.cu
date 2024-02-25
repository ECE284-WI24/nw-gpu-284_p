#ifndef NW_CUH
#include <iostream>
#include <stdio.h>
#include "nw.cuh"
#endif


void printGpuProperties () {
    int nDevices;

    // Store the number of available GPU device in nDevicess
    cudaError_t err = cudaGetDeviceCount(&nDevices);

    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaGetDeviceCount failed!\n");
        exit(1);
    }

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device memory: %lu\n", prop.totalGlobalMem);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}


void NWGPU::DeviceArrays::allocateDeviceArrays(
    size_t h_numAlignments,
    std::vector<char *>& h_refSeq,
    std::vector<char *>& h_querySeq,
    std::vector<size_t>& h_refLen,
    std::vector<size_t>& h_queryLen,
    Params& param
){

    cudaError_t err;

    d_numAlignments = h_numAlignments;
    d_match = param.match;
    d_mismatch = param.mismatch;
    d_gapOpen = param.gapOpen;

    
    size_t * refLen = new size_t[h_numAlignments];
    size_t * queryLen = new size_t[h_numAlignments];

    size_t * refStartCord = new size_t[h_numAlignments];
    size_t * queryStartCord = new size_t[h_numAlignments];


    size_t totalRefLength = 0;
    size_t totalQueryLength = 0;

    // Allocate Ref and Query Length memory
    err = cudaMalloc(&d_refLen, h_numAlignments*sizeof(size_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_queryLen, h_numAlignments*sizeof(size_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // Transfer Sequence Length values

    refStartCord[0]=0; queryStartCord[0]=0;
    for (size_t i=0; i<h_numAlignments; i++)
    {
        refLen[i] = h_refLen[i];
        queryLen[i] = h_queryLen[i];
        totalRefLength += h_refLen[i];
        totalQueryLength += h_queryLen[i];
        if (i < h_numAlignments - 1)
        {
            refStartCord[i+1] = totalRefLength;
            queryStartCord[i+1] = totalQueryLength;
        }

    }


    err = cudaMemcpy(d_refLen, refLen, h_numAlignments*sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_queryLen, queryLen, h_numAlignments*sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    // Allocate Start Coordinate Memory
    err = cudaMalloc(&d_refStartCord, h_numAlignments*sizeof(size_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_queryStartCord, h_numAlignments*sizeof(size_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // Transfer Start Coordinate  Data
    err = cudaMemcpy(d_refStartCord, refStartCord, h_numAlignments*sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_queryStartCord, queryStartCord, h_numAlignments*sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }


    // Allocate Ref and Query Sequence memory
    err = cudaMalloc(&d_refSeq, totalRefLength*sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    err = cudaMalloc(&d_querySeq, totalQueryLength*sizeof(char));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // Transfer Sequences
    char * ref = new char[totalRefLength];
    char * query = new char[totalQueryLength];

    uint32_t refIndex=0, queryIndex=0;

    for (size_t i=0; i<h_numAlignments; i++)
    {
        char * currentRef =  h_refSeq[i];
        char * currentQuery =  h_querySeq[i];
        size_t currentRefLength = h_refLen[i];
        size_t currentQueryLength = h_queryLen[i];
        for (size_t j=0; j<currentRefLength; j++)
            ref[refIndex++] = currentRef[j];
        for (size_t j=0; j<currentQueryLength; j++)
            query[queryIndex++] = currentQuery[j];
    }

    err = cudaMemcpy(d_refSeq, ref, totalRefLength*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    err = cudaMemcpy(d_querySeq, query, totalQueryLength*sizeof(char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }


    // Allocate memory for the output
    err = cudaMalloc(&d_scores, h_numAlignments*sizeof(uint32_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // Allocate memory to store traceback pointers
    err = cudaMalloc(&d_tbPointers, h_numAlignments*(512)*sizeof(int8_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // Allocate memory to store traceback Length
    err = cudaMalloc(&d_tbPointersLen, h_numAlignments*sizeof(uint32_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Gpu_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    cudaDeviceSynchronize();

}

void NWGPU::DeviceArrays::deallocateDeviceArrays()
{
    cudaFree(d_refSeq);
    cudaFree(d_querySeq);
    cudaFree(d_refLen);
    cudaFree(d_queryLen);
    cudaFree(d_refStartCord);
    cudaFree(d_queryStartCord);
    cudaFree(d_tbPointers);
    cudaFree(d_tbPointersLen);
}


__device__ void tracebackSeqtoSeq
(
    int8_t tbMatrix[],
    int32_t tbIdx,
    int32_t wfLL[],
    int32_t wfLen[],
    size_t refLen,
    size_t queryLen,
    int8_t * tbPointers,
    int32_t * tbPointersLen
){
    int32_t refIndex = refLen-1;
    int32_t queryIndex = queryLen-1;

    tbIdx--;
    
    int32_t k=refLen+queryLen;

    int8_t state;
    int32_t currentTbPointersIdx = 0;


    while (k>=0)
    {
        state = tbMatrix[tbIdx] & 0x03;
        tbIdx -= (refIndex - wfLL[k] + 1 + wfLen[k-1]);

        tbPointers[currentTbPointersIdx++] = state;
        if (state == 0) 
        {
            tbIdx -= wfLen[k-2]; tbIdx += refIndex - wfLL[k-2]; k--;k--;
        }
        else if (state == 1) 
        {
            tbIdx += (refIndex - wfLL[k-1] + 1); k--;
        }
        else
        {
            tbIdx += (refIndex - wfLL[k-1]); k--; 
        }
    }

    tbPointersLen[0] = currentTbPointersIdx;

}

__global__ void alignSeqToSeq
(
    size_t d_numAlignments,
    char* d_ref,
    char* d_query,
    size_t * refLen,
    size_t * queryLen,
    size_t * refStartCord,
    size_t * queryStartCord,
    int matchPoints, 
    int mismatchPoints,
    int gapOpenPoints,
    int * d_scores,
    int8_t * tbPointers,
    int32_t * tbPointersLen
){

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int bs = blockDim.x;
    int gs = gridDim.x;


    
    if (tx==0 and bx==0)
    {
        for (size_t n=0; n<d_numAlignments; n++)
        {
            size_t currentRefLength = refLen[n];
            size_t currentQueryLength = queryLen[n];

            size_t currentRefStartCord = refStartCord[n];
            size_t currentQueryStartCord = queryStartCord[n];

            char * ref = d_ref + currentRefStartCord;
            char * query = d_query + currentQueryStartCord;
            
            int8_t * currentTbPointers = tbPointers + n*512;
            int32_t * currentTbPointersLen = tbPointersLen + n;

            size_t maxWFLen = currentRefLength + currentQueryLength + 2; //wavefront length

            int32_t score = 0;
            int32_t maxScore = 0;

            int32_t H[3][500];
            int32_t L[3], U[3];

            int32_t wfLL[256*2+2];
            int32_t wfLen[256*2+2];
            int8_t tbMatrix[258*258]; //(256+2)^2
            int32_t tbIdx = 0;

            int8_t state=0;

            for(size_t i=0; i<3; i++)
            {
                L[i]=0; U[i]=0;
            }

            for (size_t i=0; i<3; i++)
            {
                for (size_t j=0; j<500; j++) H[i][j] = 0;
            }

            /* k -> Antidiagonal Index */
            for (int32_t k=0; k<currentRefLength+currentQueryLength+1; k++)
            {
                L[k%3] = (k<=currentQueryLength)?0:k-currentQueryLength;
                U[k%3] = (k<=currentRefLength)?k:currentRefLength;
                wfLL[k] = L[k%3];
                wfLen[k] = U[k%3]-L[k%3]+1;

                for(int32_t i=L[k%3]; i<U[k%3]+1; i++) // i -> Reference Index
                {   
                    int32_t j=(k-i); //j->Query Index
                    int32_t match = -INF, insOp = -INF, delOp = -INF;
                    int32_t offset = i-L[k%3];
                    int32_t offsetDiag = L[k%3]-L[(k+1)%3]+offset-1;
                    int32_t offsetUp = L[k%3]-L[(k+2)%3]+offset;
                    int32_t offsetLeft = L[k%3]-L[(k+2)%3]+offset-1;


                    if (k==0) match = 0;
                    
                    if (offsetDiag>=0)
                    {
                        char refVal = ref[i-1];
                        char queryVal = query[j-1];
                        if (refVal == queryVal) match = H[(k+1)%3][offsetDiag] + matchPoints;
                        else match = H[(k+1)%3][offsetDiag] + mismatchPoints;
                    }
                    
                    if (offsetUp >= 0)
                        insOp = H[(k+2)%3][offsetUp] + gapOpenPoints;

                    if (offsetLeft >=0)
                        delOp = H[(k+2)%3][offsetLeft] + gapOpenPoints;

                    
                    if (match > insOp) 
                    {
                        if (match > delOp) 
                        {
                            H[k%3][offset] = match;
                            state = 0;
                        }
                        else 
                        {
                            H[k%3][offset] = delOp;
                            state = 1;
                        }
                    }
                    else if (insOp > delOp) 
                    {
                        H[k%3][offset] = insOp;
                        state = 1;
                    }
                    else 
                    {
                        H[k%3][offset] = delOp;
                        state = 2;
                    }

                    tbMatrix[tbIdx++] = state;

                    score = H[k%3][offset];
                    if (score > maxScore) maxScore = score;
                }
            }
            
            d_scores[n] = score;
            tracebackSeqtoSeq(tbMatrix,tbIdx, wfLL, wfLen, currentRefLength, currentQueryLength, currentTbPointers, currentTbPointersLen);
        }
        
    }

}



void NWGPU::NWonGPU
(
    size_t d_numAlignments,
    char * d_ref,
    char * d_query,
    size_t * d_refLen,
    size_t * d_queryLen,
    size_t * d_refStartCord,
    size_t * d_queryStartCord,
    int d_match,
    int d_mismatch,
    int d_gapOpen,
    int * d_scores,
    int8_t * d_tbPointers,
    int32_t * d_tbPointersLen
){

    int blockPerGrid = 1;
    int threadsPerBlock = 1;


    alignSeqToSeq<<<blockPerGrid,threadsPerBlock>>>( d_numAlignments,
                            d_ref,
                            d_query,
                            d_refLen,
                            d_queryLen,
                            d_refStartCord,
                            d_queryStartCord,
                            d_match,
                            d_mismatch,
                            d_gapOpen,
                            d_scores,
                            d_tbPointers,
                            d_tbPointersLen);

    cudaDeviceSynchronize();

}

void NWGPU::DeviceArrays::printScores(size_t h_numAlignments)
{
    cudaError_t err;
    int* h_scores = new int[h_numAlignments];

    err = cudaMemcpy(h_scores, d_scores, h_numAlignments*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    printf("Count\tScore\n");
    for (size_t i=0; i<h_numAlignments; i++)
    {
        printf("%zu\t%d\n",i,h_scores[i]);
    }
}

void NWGPU::DeviceArrays::printTbPointers(size_t h_numAlignments)
{
    cudaError_t err;
    int8_t* tbPointers = new int8_t[h_numAlignments*512];
    int32_t* tbPointersLen = new int32_t[h_numAlignments];

    err = cudaMemcpy(tbPointers, d_tbPointers, h_numAlignments*512*sizeof(int8_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    err = cudaMemcpy(tbPointersLen, d_tbPointersLen, h_numAlignments*sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    for (size_t i=0; i<h_numAlignments; i++)
    {
        int8_t * currentTbPointers = tbPointers + 512*i;
        printf ("%d\t", i);
        for (size_t j=0; j<tbPointersLen[i]; j++)
        {
            printf("%d", currentTbPointers[j]);
        }
        printf("\n");
    }
}


