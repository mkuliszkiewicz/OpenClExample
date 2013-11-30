
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <CL/cl.h>
#include <string.h>
#include <stdint.h>

#define DEBUG
#define DIM 1792
#define SIZE DIM * DIM

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
   int i;
   for (i = 0; i < size; ++i)
   data[i] = rand() / (float)RAND_MAX;
}

void checkRet(cl_int *ret, const char * itemName)
{
  if (*ret != CL_SUCCESS)
  {
      printf("ERROR %s: %d\n", itemName, *ret);
      // exit(-1);
  }
}

void get_avb_threads_count(int wanted, const int max_threads, size_t *ret) 
{
  size_t no_threads = 1;

  if (wanted > max_threads) wanted = max_threads;

  while (no_threads * no_threads <= wanted) {
    if (no_threads * 2 * no_threads * 2 <= wanted)
      no_threads *= 2;
    else
      break;
  }

  ret[0] = 8;
  ret[1] = 8;

#ifdef DEBUG
  printf("wanted %d got %zu\n", wanted,no_threads * no_threads);
#endif
}

void printMatrix(float *matrix, int matrixSize)
{
  int i;
  for(i = 0; i < matrixSize; i++)
  {
    printf("%f ", matrix[i]);
    if(((i + 1) % SIZE) == 0)
    printf("\n");
  }
}

void checkGPUResults(float *A, float *B, float *C, int matrixDim)
{
    int ii,jj, kk;
    int matrixValid = 1;  
    for (ii = 0; ii < matrixDim; ii++) // row
    {
      for (jj = 0; jj < matrixDim; jj++) // col
      {
        float ssum = 0.0;
        for (kk = 0; kk < matrixDim; kk++) // col
        {
           ssum += A[ii * matrixDim + kk] * B[kk * matrixDim + jj];
        }
        if(C[ii * matrixDim + jj] - ssum > FLT_EPSILON)
        {
          printf("-> Invalid (%d %d)\n", ii, jj);
          matrixValid = 0;
        } 
      }
    }
    if (matrixValid == 1) printf("-> No errors\n");
}

void printMatrixWithName(float *matrix, int matrixSize, const char *matrixName)
{
  printf("----------------\n");
  printf("%s\n", matrixName);
  printMatrix(matrix, matrixSize);
  printf("----------------\n");
}


int main(int argc, char** argv)
{
  if (argc < 3)
  {
    printf("Bad usage <num_threads> <matrix_size>\n");
    exit(-1);
  }

  int mDIM = atoi(argv[2]);
  int clThreadsNum = atoi(argv[1]);
  int mSIZE = mDIM * mDIM;
  
  
  size_t th[2];
  get_avb_threads_count(clThreadsNum, 512, th);
  srand(time(NULL));
  
  // Loop index used in whole function
  int i;

  unsigned int size_A = mSIZE;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A = (float*) malloc(mem_size_A);

  unsigned int size_B = mSIZE;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B = (float*) malloc(mem_size_B);
  float* checkMatrix = (float*) malloc(mem_size_B);

  unsigned int size_C = mSIZE;
  unsigned int mem_size_C = sizeof(float) * size_C;
  float* h_C = (float*) malloc(mem_size_C);

  #ifdef DEBUG
    printf("TASK SIZE: %lf [MB]\n", 3.0f * (float)mSIZE * (float)sizeof(float) / 1000000.0f);
  #endif

  randomInit(h_A, mSIZE);
  randomInit(h_B, mSIZE);
 

  //////////////////////////////
  /// OpenCL specific variables
  cl_context clGPUContext;// computation context
  cl_command_queue clCommandQueue; // command queue
  cl_program clProgram; // opencl program
  cl_kernel clKernel; // opencl kernel
  cl_platform_id clPlatformId;
  cl_device_id clDevise;
  cl_uint clNumDevices;
  cl_uint clNumPlatforms;
  cl_int errcode;
  cl_build_status status;
  cl_event clMeasureEvent;

  //////////////////////////////
  /// Matrices memmory
  cl_mem d_A;
  cl_mem d_B;
  cl_mem d_C;

  //////////////////////////////
  /// OpenCL setup
  errcode = clGetPlatformIDs(1, &clPlatformId, &clNumPlatforms);
  checkRet(&errcode, "clGetPlatformIDs"); 

  errcode = clGetDeviceIDs(clPlatformId, CL_DEVICE_TYPE_GPU, 1, &clDevise, &clNumDevices);
  checkRet(&errcode, "clGetDeviceIDs");

  clGPUContext = clCreateContext(NULL, 1, &clDevise, NULL, NULL, &errcode);
  checkRet(&errcode, "clCreateContext");

  clCommandQueue = clCreateCommandQueue(clGPUContext, clDevise, CL_QUEUE_PROFILING_ENABLE, &errcode);
  checkRet(&errcode, "clCreateCommandQueue");
  

  size_t clMatrixMulSize;
  char *clMatrixMul = 
  "__kernel void matrixMul(__global float* C, \n" \
    "                       __global float* A, \n" \
  "                       __global float* B, \n" \
  "                                  int dim) \n" \
  "    {\n" \
  "         int tx = get_global_id(0); \n" \
  "         int ty = get_global_id(1); \n" \
  "         float value = 0; \n"\
         "for (int k = 0; k < dim; ++k) \n" \
         "{" \
           " float elementA = A[ty * dim + k]; \n" \
                       "float elementB = B[k * dim + tx]; \n" \
                       "value += elementA * elementB; \n" \
         "} \n" \
         "C[ty * dim + tx] = value; \n" \
    "}";

  clMatrixMulSize = strlen(clMatrixMul);

  clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&clMatrixMul, NULL, &errcode);
  if (errcode != 0)
  {
      printf("ERROR Creating program: %d\n", errcode);
  }

   errcode = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
   checkRet(&errcode, "clBuildProgram");
   // if (errcode != CL_SUCCESS) 
   // {
   //    size_t logSize;
   //    char *programLog;
   //    // check build error and build status first
   //    clGetProgramBuildInfo(clProgram, clDevise, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);

   //    // check build log
   //    clGetProgramBuildInfo(clProgram, clDevise, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
   //    programLog = (char*) calloc (logSize+1, sizeof(char));
   //    clGetProgramBuildInfo(clProgram, clDevise, CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
   //    printf("Build failed; error=%d, status=%d, programLog:\n\n%s", errcode, status, programLog);
   //    free(programLog);
   //  }
 
  clKernel = clCreateKernel(clProgram, "matrixMul", &errcode);
  checkRet(&errcode, "clCreateKernel");

  d_C = clCreateBuffer(clGPUContext, CL_MEM_WRITE_ONLY, mem_size_A, NULL, &errcode);
  checkRet(&errcode, "clCreateBuffer C");

  d_A = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, mem_size_A, h_A, &errcode);
  checkRet(&errcode, "clCreateBuffer A");

  d_B = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, mem_size_B, h_B, &errcode);
  checkRet(&errcode, "clCreateBuffer B");
 
  errcode = clEnqueueWriteBuffer(clCommandQueue, d_A, CL_TRUE, 0, mSIZE * sizeof(float), h_A, 0, NULL, NULL);
  checkRet(&errcode, "clEnqueueWriteBuffer");

  errcode = clEnqueueWriteBuffer(clCommandQueue, d_B, CL_TRUE, 0, mSIZE * sizeof(float), h_B, 0, NULL, NULL);
  checkRet(&errcode, "clEnqueueWriteBuffer");

  

  ///////////////////////////
  // Kernel arguments
  errcode = clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&d_C);
  errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&d_A);
  errcode |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&d_B);
  errcode |= clSetKernelArg(clKernel, 3, sizeof(int), (void *)&mDIM);
  checkRet(&errcode, "clSetKernelArg");

  ///////////////////////////
  // Number of work-items
  size_t localWorkSize[2], globalWorkSize[2];
  localWorkSize[0] = th[0];
  localWorkSize[1] = th[1];
  globalWorkSize[0] = mDIM;
  globalWorkSize[1] = mDIM;

  errcode = clFinish(clCommandQueue);
  checkRet(&errcode, "clFinish [BEFORE enqueueing]");

  errcode = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &clMeasureEvent);
  checkRet(&errcode, "clEnqueueNDRangeKernel");

  //////////////////////////////
  /// Wait for enqueueing and measure time

  errcode = clWaitForEvents(1, &clMeasureEvent);
  checkRet(&errcode, "clWaitForEvents");
  
  long long start, end;
  double total;

  errcode = clGetEventProfilingInfo(clMeasureEvent, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
  checkRet(&errcode, "clGetEventProfilingInfo");

  errcode = clGetEventProfilingInfo(clMeasureEvent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
  checkRet(&errcode, "clGetEventProfilingInfo");

  total = (double)(end - start) / 1e6; // ns -> ms
  printf("%5.2f\n", total);



  //////////////////////////////
  /// Read results
  errcode = clEnqueueReadBuffer(clCommandQueue, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);
  checkRet(&errcode, "clEnqueueReadBuffer");
   
  //////////////////////////////
  /// Print results and calculate on CPU
  #ifdef DEBUG    
  // printMatrixWithName(h_C, SIZE, "Matrix C (FROM GPU)");
  checkGPUResults(h_A, h_B, h_C, mDIM); 
  #endif
  
  //////////////////////////////
  /// Tear down
  free(h_A);
  free(h_B);
  free(h_C);
  clFlush(clCommandQueue);
  clReleaseMemObject(d_A);
  clReleaseMemObject(d_C);
  clReleaseMemObject(d_B);      
  clReleaseKernel(clKernel);
  clReleaseProgram(clProgram);
  clReleaseCommandQueue(clCommandQueue);
  clReleaseContext(clGPUContext);
  clReleaseEvent(clMeasureEvent);

  return 0;
}