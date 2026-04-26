
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// ── GPU Kernel: RGB to Grayscale ─────────────────────────────────────────────
__global__ void convertRGBToGrayscale(
    unsigned char* input,
    unsigned char* output,
    int rows,
    int columns)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < columns)
    {
        int idx = row * columns + col;
        unsigned char r = input[3 * idx + 0];
        unsigned char g = input[3 * idx + 1];
        unsigned char b = input[3 * idx + 2];
        // ITU-R BT.709 luminance formula
        output[idx] = (unsigned char)(0.2126f * r + 0.7152f * g + 0.0722f * b);
    }
}

// ── GPU Kernel: Gaussian Blur (3x3) ─────────────────────────────────────────
__global__ void applyGaussianBlur(
    unsigned char* input,
    unsigned char* output,
    int rows,
    int columns)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row > 0 && row < rows - 1 && col > 0 && col < columns - 1)
    {
        // 3x3 Gaussian kernel weights
        float kernel[3][3] = {
            {1/16.0f, 2/16.0f, 1/16.0f},
            {2/16.0f, 4/16.0f, 2/16.0f},
            {1/16.0f, 2/16.0f, 1/16.0f}
        };

        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                int idx = (row + ky) * columns + (col + kx);
                sum += input[idx] * kernel[ky + 1][kx + 1];
            }
        }
        output[row * columns + col] = (unsigned char)sum;
    }
    else if (row < rows && col < columns)
    {
        output[row * columns + col] = input[row * columns + col];
    }
}

// ── CPU Reference: RGB to Grayscale ─────────────────────────────────────────
void cpuConvertRGBToGrayscale(
    unsigned char* input,
    unsigned char* output,
    int rows,
    int columns)
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < columns; col++)
        {
            int idx = row * columns + col;
            unsigned char r = input[3 * idx + 0];
            unsigned char g = input[3 * idx + 1];
            unsigned char b = input[3 * idx + 2];
            output[idx] = (unsigned char)(0.2126f * r + 0.7152f * g + 0.0722f * b);
        }
    }
}

// ── Process single image on GPU ───────────────────────────────────────────────
double processImageGPU(
    unsigned char* h_input,
    unsigned char* h_output,
    int rows, int columns,
    int applyBlur)
{
    int rgbSize  = rows * columns * 3;
    int graySize = rows * columns;

    unsigned char *d_input, *d_gray, *d_blurred;
    cudaMalloc(&d_input,   rgbSize);
    cudaMalloc(&d_gray,    graySize);
    cudaMalloc(&d_blurred, graySize);

    cudaMemcpy(d_input, h_input, rgbSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((columns + 15) / 16, (rows + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Step 1: Convert to grayscale
    convertRGBToGrayscale<<<gridSize, blockSize>>>(d_input, d_gray, rows, columns);

    // Step 2: Optional Gaussian blur
    if (applyBlur)
    {
        applyGaussianBlur<<<gridSize, blockSize>>>(d_gray, d_blurred, rows, columns);
        cudaMemcpy(h_output, d_blurred, graySize, cudaMemcpyDeviceToHost);
    }
    else
    {
        cudaMemcpy(h_output, d_gray, graySize, cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blurred);

    return (double)elapsed;
}

// ── Process single image on CPU ───────────────────────────────────────────────
double processImageCPU(
    unsigned char* h_input,
    unsigned char* h_output,
    int rows, int columns)
{
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);

    cpuConvertRGBToGrayscale(h_input, h_output, rows, columns);

    clock_gettime(CLOCK_MONOTONIC, &stop);

    double elapsed = (stop.tv_sec - start.tv_sec) * 1000.0
                   + (stop.tv_nsec - start.tv_nsec) / 1e6;
    return elapsed;
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    // Default arguments
    const char* inputDir  = "data/input";
    const char* outputDir = "data/output";
    const char* logPath   = "data/results.csv";
    int applyBlur = 0;

    // Parse CLI arguments
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-i") && i+1 < argc) inputDir  = argv[++i];
        if (!strcmp(argv[i], "-o") && i+1 < argc) outputDir = argv[++i];
        if (!strcmp(argv[i], "-l") && i+1 < argc) logPath   = argv[++i];
        if (!strcmp(argv[i], "-b"))                applyBlur = 1;
    }

    printf("================================================\n");
    printf("  CUDA GPU Image Processing Pipeline\n");
    printf("================================================\n");
    printf("Input:      %s\n", inputDir);
    printf("Output:     %s\n", outputDir);
    printf("Log:        %s\n", logPath);
    printf("Blur:       %s\n", applyBlur ? "enabled" : "disabled");
    printf("================================================\n\n");

    // Print GPU info
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU %d: %s\n", i, prop.name);
        printf("  Compute: %d.%d\n", prop.major, prop.minor);
        printf("  Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    }
    printf("\n");

    // Open log file
    FILE* logFile = fopen(logPath, "w");
    fprintf(logFile, "filename,rows,columns,pixels,gpu_ms,cpu_ms,speedup\n");

    // Read manifest
    char manifestPath[512];
    snprintf(manifestPath, sizeof(manifestPath), "%s/manifest.txt", inputDir);
    FILE* manifest = fopen(manifestPath, "r");
    if (!manifest)
    {
        printf("ERROR: No manifest.txt in %s\n", inputDir);
        return 1;
    }

    char filename[256];
    int processed = 0;
    double totalGPU = 0.0, totalCPU = 0.0;

    while (fgets(filename, sizeof(filename), manifest))
    {
        filename[strcspn(filename, "\n")] = 0;
        if (!strlen(filename)) continue;

        char inputPath[512], outputPath[512];
        snprintf(inputPath,  sizeof(inputPath),  "%s/%s", inputDir,  filename);
        snprintf(outputPath, sizeof(outputPath), "%s/%s_out.png", outputDir, filename);

        // Load image
        int width, height, channels;
        unsigned char* img = stbi_load(inputPath, &width, &height, &channels, 3);
        if (!img)
        {
            printf("SKIP: %s\n", filename);
            continue;
        }

        int graySize = height * width;
        unsigned char* h_gpu_out = (unsigned char*)malloc(graySize);
        unsigned char* h_cpu_out = (unsigned char*)malloc(graySize);

        // GPU processing
        double gpuTime = processImageGPU(img, h_gpu_out, height, width, applyBlur);

        // CPU processing (for comparison)
        double cpuTime = processImageCPU(img, h_cpu_out, height, width);

        double speedup = cpuTime / gpuTime;

        // Save output
        stbi_write_png(outputPath, width, height, 1, h_gpu_out, width);

        printf("[%3d] %-30s %4dx%-4d  GPU: %7.3f ms  CPU: %7.3f ms  Speedup: %.2fx\n",
               processed + 1, filename, height, width, gpuTime, cpuTime, speedup);

        fprintf(logFile, "%s,%d,%d,%d,%.4f,%.4f,%.4f\n",
                filename, height, width, height*width, gpuTime, cpuTime, speedup);

        stbi_image_free(img);
        free(h_gpu_out);
        free(h_cpu_out);

        totalGPU += gpuTime;
        totalCPU += cpuTime;
        processed++;
    }

    fclose(manifest);
    fclose(logFile);

    printf("\n================================================\n");
    printf("  RESULTS SUMMARY\n");
    printf("================================================\n");
    printf("  Images processed : %d\n", processed);
    printf("  Total GPU time   : %.2f ms\n", totalGPU);
    printf("  Total CPU time   : %.2f ms\n", totalCPU);
    printf("  Avg GPU/image    : %.4f ms\n", totalGPU / processed);
    printf("  Avg CPU/image    : %.4f ms\n", totalCPU / processed);
    printf("  Avg Speedup      : %.2fx\n",   totalCPU / totalGPU);
    printf("================================================\n");

    cudaDeviceReset();
    return 0;
}
