#include <iostream>
#include <stdio.h>
#include <vector>
#include <boost/program_options.hpp>
#include <tbb/task_scheduler_init.h>
#include "kseq.h"
#include "zlib.h"
#include "timer.hpp"
 
#include "nw.cuh"
 
 
// For parsing the command line values
namespace po = boost::program_options;
 
// For reading in the FASTA file
KSEQ_INIT2(, gzFile, gzread)
void cpu_tracebackSeqtoSeq(
    int8_t tbMatrix[],
    int32_t tbIdx,
    int32_t wfLL[],
    int32_t wfLen[],
    size_t refLen,
    size_t queryLen,
    int8_t * tbPointers,
    int32_t * tbPointersLen
    )
{
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
 
int main(int argc, char** argv) {
    // Timer below helps with the performance profiling (see timer.hpp for more
    // details)
    Timer timer;
 
    int n;
    gzFile fp;
    kseq_t *kseq_rd;
    uint32_t * twoBitCompressed;
    uint32_t numThreads;
    std::string refFilename, queryFilename;
    std::vector<char *> ref, query;
    std::vector<size_t> refLen, queryLen;
 
    // Parse the command line options
    po::options_description desc{"Options"};
    desc.add_options()
    ("ref,r", po::value<std::string>(&refFilename)->required(), "Reference FASTA file name [REQUIRED].")
    ("query,q", po::value<std::string>(&queryFilename)->required(), "Query FASTA file name [REQUIRED].")
    ("numThreads,T", po::value<uint32_t>(&numThreads)->default_value(1), "Number of Threads (range: 1-8)")
    ("help,h", "Print help messages");
 
    po::options_description allOptions;
    allOptions.add(desc);
 
    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(allOptions).run(), vm);
        po::notify(vm);
    } catch(std::exception &e) {
        std::cerr << desc << std::endl;
        exit(1);
    }
 
    if ((numThreads < 1) || (numThreads > 8)) {
        std::cerr << "ERROR! numThreads should be between 1 and 8." << std::endl;
        exit(1);
    }
 
    // Print GPU information
    timer.Start();
    fprintf(stdout, "Setting CPU threads to %u and printing GPU device properties.\n", numThreads);
    tbb::task_scheduler_init init(numThreads);
    printGpuProperties();
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());
 
    // Read reference sequence
    timer.Start();
    fprintf(stdout, "Reading reference sequence\n");
    fp = gzopen(refFilename.c_str(), "r");
    if (!fp) {
        fprintf(stdout, "ERROR: Cannot open file: %s\n", refFilename.c_str());
        exit(1);
    }
    kseq_rd = kseq_init(fp);
 
    while (kseq_read(kseq_rd) >= 0) {
        size_t seqLen = kseq_rd->seq.l;
        // std::string seqString = std::string(kseq_rd->seq.s, seqLen);
        
        // uint32_t twoBitCompressedSize = (seqLen+15)/16;
        // twoBitCompressed = new uint32_t[twoBitCompressedSize];
        // twoBitCompress(kseq_rd->seq.s, seqLen, twoBitCompressed);
 
        char * refSeq = new char[seqLen];
        strcpy(refSeq,kseq_rd->seq.s);
        refLen.push_back(seqLen);
        ref.push_back(refSeq);
    }
 
    if (ref.size() == 0) 
    {
        fprintf(stdout, "ERROR: No reference sequence found!\n");
        exit(1);
    }
 
    // Read query sequence
    timer.Start();
    fprintf(stdout, "Reading query sequence\n");
    fp = gzopen(queryFilename.c_str(), "r");
    if (!fp) {
        fprintf(stdout, "ERROR: Cannot open file: %s\n", queryFilename.c_str());
        exit(1);
    }
    kseq_rd = kseq_init(fp);
 
    while (kseq_read(kseq_rd) >= 0) {
        size_t seqLen = kseq_rd->seq.l;
	//printf("skoparkar debug name : %s\n", kseq_rd->name.s);
        // std::string seqString = std::string(kseq_rd->seq.s, seqLen);
 
        // uint32_t twoBitCompressedSize = (seqLen+15)/16;
        // twoBitCompressed = new uint32_t[twoBitCompressedSize];
        // twoBitCompress(kseq_rd->seq.s, seqLen, twoBitCompressed);
 
        char * querySeq = new char[seqLen];
        strcpy(querySeq,kseq_rd->seq.s);
        queryLen.push_back(seqLen);
        query.push_back(querySeq);
    }
 
    if (query.size() == 0) 
    {
        fprintf(stdout, "ERROR: Cannot open file: %s\n", queryFilename.c_str());
        exit(1);
    }
 
    // Error if reference and query sequence count mismatch
    if (ref.size() != query.size())
    {
        fprintf(stderr, "Error: reference and query sequence count mismatch (%ld,%ld)\n", ref.size(), query.size());
    }
 
    size_t numAlignments = ref.size();
 
    // Setting alignment parameters
    Params param(2,-1,-1);
    std::pair<std::string, std::string> alignment;
 
 
    // Create arrays
    timer.Start();
    fprintf(stdout, "\nAllocating GPU device arrays.\n");
    //NWGPU::deviceArrays.allocateDeviceArrays(numAlignments, ref, query, refLen, queryLen, param);
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());
 
    
    // CPU impl
    
    //NWGPU::NWonGPU(NWGPU::deviceArrays.d_numAlignments,
    //        NWGPU::deviceArrays.d_refSeq,
    //        NWGPU::deviceArrays.d_querySeq,
    //        NWGPU::deviceArrays.d_refLen,
    //        NWGPU::deviceArrays.d_queryLen,
    //        NWGPU::deviceArrays.d_refStartCord,
    //        NWGPU::deviceArrays.d_queryStartCord,
    //        NWGPU::deviceArrays.d_match,
    //        NWGPU::deviceArrays.d_mismatch,
    //        NWGPU::deviceArrays.d_gapOpen,
    //        NWGPU::deviceArrays.d_scores,
    //        NWGPU::deviceArrays.d_tbPointers,
    //        NWGPU::deviceArrays.d_tbPointersLen
    //    );
    
    size_t * refStartCord = new size_t[numAlignments];
    size_t * queryStartCord = new size_t[numAlignments];
    int8_t* tbPointers = new int8_t[numAlignments*512];
    int32_t* tbPointersLen = new int32_t[numAlignments];
    int* scores = new int[numAlignments];
    int matchPoints = param.match;
    int mismatchPoints = param.mismatch;
    int gapOpenPoints = param.gapOpen;
 
    size_t totalRefLength = 0;
    size_t totalQueryLength = 0;
 
    refStartCord[0]=0; queryStartCord[0]=0;
    for (size_t i=0; i<numAlignments; i++)
    {
        refLen[i] = refLen[i];
        queryLen[i] = queryLen[i];
        totalRefLength += refLen[i];
        totalQueryLength += queryLen[i];
        if (i < numAlignments - 1)
        {
            refStartCord[i+1] = totalRefLength;
            queryStartCord[i+1] = totalQueryLength;
        }
 
    }
 
    char * d_ref = new char[totalRefLength];
    char * d_query = new char[totalQueryLength];
 
    uint32_t refIndex=0, queryIndex=0;
 
    for (size_t i=0; i<numAlignments; i++)
    {
        char * currentRef =  ref[i];
        char * currentQuery =  query[i];
        size_t currentRefLength = refLen[i];
        size_t currentQueryLength = queryLen[i];
        for (size_t j=0; j<currentRefLength; j++)
            d_ref[refIndex++] = currentRef[j];
        for (size_t j=0; j<currentQueryLength; j++)
            d_query[queryIndex++] = currentQuery[j];
    }
 
    for (size_t n=0; n<numAlignments; n++)
        {
            size_t currentRefLength = refLen[n];
            size_t currentQueryLength = queryLen[n];
 
            size_t currentRefStartCord = refStartCord[n];
            size_t currentQueryStartCord = queryStartCord[n];
 
            char * current_ref = d_ref + currentRefStartCord;
            char * current_query = d_query + currentQueryStartCord;
            
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
                        char refVal = current_ref[i-1];
                        char queryVal = current_query[j-1];
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
            
            scores[n] = score;
            cpu_tracebackSeqtoSeq(tbMatrix,tbIdx, wfLL, wfLen, currentRefLength, currentQueryLength, currentTbPointers, currentTbPointersLen);
        }
 
    // Print Scores
    //NWGPU::deviceArrays.printScores(numAlignments);
    printf("Count\tScore\n");
    for (size_t i=0; i<numAlignments; i++)
    {
        printf("%zu\t%d\n",i,scores[i]);
    }
    //NWGPU::deviceArrays.printTbPointers(numAlignments);
    for (size_t i=0; i<numAlignments; i++)
    {
        int8_t * currentTbPointers = tbPointers + 512*i;
        printf ("%d\t", i);
        for (size_t j=0; j<tbPointersLen[i]; j++)
        {
            printf("%d", currentTbPointers[j]);
        }
        printf("\n");
    }
 
    // Delete arrays
    timer.Start();
    fprintf(stdout, "Deallocating CPU and GPU arrays.\n");
    NWGPU::deviceArrays.deallocateDeviceArrays();
    delete [] twoBitCompressed;
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());
 
    return 0;
}