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
    NWGPU::deviceArrays.allocateDeviceArrays(numAlignments, ref, query, refLen, queryLen, param);
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    
    NWGPU::NWonGPU(NWGPU::deviceArrays.d_numAlignments,
            NWGPU::deviceArrays.d_refSeq,
            NWGPU::deviceArrays.d_querySeq,
            NWGPU::deviceArrays.d_refLen,
            NWGPU::deviceArrays.d_queryLen,
            NWGPU::deviceArrays.d_refStartCord,
            NWGPU::deviceArrays.d_queryStartCord,
            NWGPU::deviceArrays.d_match,
            NWGPU::deviceArrays.d_mismatch,
            NWGPU::deviceArrays.d_gapOpen,
            NWGPU::deviceArrays.d_scores,
            NWGPU::deviceArrays.d_tbPointers,
            NWGPU::deviceArrays.d_tbPointersLen
        );
    
    // Print Scores
    NWGPU::deviceArrays.printScores(numAlignments);
    NWGPU::deviceArrays.printTbPointers(numAlignments);

    // Delete arrays
    timer.Start();
    fprintf(stdout, "Deallocating CPU and GPU arrays.\n");
    NWGPU::deviceArrays.deallocateDeviceArrays();
    delete [] twoBitCompressed;
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    return 0;
}
