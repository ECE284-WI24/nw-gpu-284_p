#ifndef COMMON_CUH
#define COMMON_CUH

#include <vector>
#define INF 10000

void printGpuProperties();
struct Params 
{
    int16_t match;
    int16_t mismatch;
    int16_t gapOpen;

    Params(int16_t t_match, int16_t t_mismatch, int16_t t_gapOpen) : //linear gap
        match(t_match), mismatch(t_mismatch), gapOpen(t_gapOpen) {}

};

namespace NWGPU
{
    struct DeviceArrays
    {       
        size_t     d_numAlignments;    
        char * d_refSeq;
        char * d_querySeq;
        size_t *    d_refLen;
        size_t *    d_queryLen;
        size_t *    d_refStartCord;
        size_t *    d_queryStartCord;

        int d_match;
        int d_mismatch;
        int d_gapOpen;
        int * d_scores;
        int8_t * d_tbPointers;
        int32_t * d_tbPointersLen;
        // HINT: if needed, you add more device arrays for the GPU here (make sure to allocate and dellocate them in appropriate functions!)

        void allocateDeviceArrays (
            size_t h_numAlignments,
            std::vector<char *>& h_refSeq,
            std::vector<char *>& h_querySeq,
            std::vector<size_t>& h_refLen,
            std::vector<size_t>& h_queryLen,
            Params& params
        );
        void deallocateDeviceArrays ();
        void printScores(size_t numAlignments);
        void printTbPointers(size_t numAlignments);
    };
    static DeviceArrays deviceArrays;

    void NWonGPU(
        size_t numAlignments,
        char * ref,
        char * query,
        size_t * refLen,
        size_t * queryLen,
        size_t * refStartCord,
        size_t * queryStartCord,
        int match,
        int mismatch,
        int gapOpen,
        int * scores,
        int8_t * tbPointers,
        int32_t * tbPointersLen
    );

}


#endif