// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Test.cu                                                     (C) 2000-2025 */
/*                                                                           */
/* Fichier contenant les tests pour l'implémentation CUDA.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/NumArray.h"
#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/Item.h"
#include "arcane/core/MathUtils.h"

#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/RunQueue.h"

#include "arcane/accelerator/cuda/CudaAccelerator.h"
#include "arcane/accelerator/RunCommandLoop.h"

using namespace Arccore;
using namespace Arcane;
using namespace Arcane::Accelerator;

__device__ __forceinline__ unsigned int getGlobalIdx_1D_1D()
{
  unsigned int blockId = blockIdx.x;
  unsigned int threadId = blockId * blockDim.x + threadIdx.x;
  return threadId;
}

template <typename T>
struct Privatizer
{
  using value_type = T; //std::decay<T>;
  using reference_type = value_type&;
  value_type priv;

  ARCCORE_HOST_DEVICE Privatizer(const T& o) : priv{o} {}
  ARCCORE_HOST_DEVICE reference_type get_priv() { return priv; }
};

template <typename T>
ARCCORE_HOST_DEVICE auto thread_privatize(const T& item) -> Privatizer<T>
{
  return Privatizer<T>{item};
}

__global__ void MyVecAdd(double* a,double* b,double* out)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  out[i] = a[i] + b[i];
  if (i<10){
    printf("A=%d %lf %lf %lf %d\n",i,a[i],b[i],out[i],3);
  }
}

__global__ void MyVecAdd2(Span<const double> a,Span<const double>b,Span<double> out)
{
  Int64 size = a.size();
  Int64 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=size)
    return;
  out[i] = a[i] + b[i];
  if (i<10){
    printf("A=%ld %lf %lf %lf %ld\n",i,a[i],b[i],out[i],i);
  }
}

__global__ void MyVecAdd3(MDSpan<const double,MDDim1> a,MDSpan<const double,MDDim1> b,MDSpan<double,MDDim1> out)
{
  int size = a.extent0();
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=size)
    return;
  out(i) = a(i) + b(i);
  if (i<10){
    printf("A=%d %lf %lf %lf %d\n",i,a(i),b(i),out(i),i);
  }
}

__global__ void
MyEmptyKernel()
{
}

void _initArrays(Span<double> a,Span<double> b,Span<double> c,int base)
{
  Int64 vsize = a.size();
  for( Int64 i = 0; i<vsize; ++i ){
    a[i] = (double)(i+base);
    b[i] = (double)(i*i+base);
    c[i] = 0.0;
  }
}

void _initArrays(MDSpan<double,MDDim1> a,MDSpan<double,MDDim1> b,MDSpan<double,MDDim1> c,int base)
{
  Int32 vsize = a.extent0();
  for( Int32 i = 0; i<vsize; ++i ){
    a(i) = (double)(i+base);
    b(i) = (double)(i*i+base);
    c(i) = 0.0;
  }
}

template<typename F> __global__ 
void MyVecLambda(int size,F func)
{
  auto privatizer = thread_privatize(func);
  auto& body = privatizer.get_priv();

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<size)
    body(i);
}

namespace TestCuda
{
class IA
{
  virtual __device__ __host__ void DoIt2() =0;
};

class A
: public IA
{
 public:
  //__global__ void DoIt(){}
  virtual __device__ __host__ void DoIt2() override {}
};
}

void MyTestFunc1()
{
  struct Context1
  {
    int a;
  };
  auto k = [=](Context1& ctx){ std::cout << "A=" << ctx.a << "\n"; };
  Context1 my_ctx;
  my_ctx.a = 3;
  k(my_ctx);
}

extern "C"
int arcaneTestCuda1()
{
  constexpr int vsize = 2000;
  std::vector<double> a(vsize);
  std::vector<double> b(vsize);
  std::vector<double> out(vsize);
  for( size_t i = 0; i<vsize; ++i ){
    a[i] = (double)(i+1);
    b[i] = (double)(i*i+1);
    out[i] = 0.0; //a[i] + b[i];
  }
  size_t mem_size = vsize*sizeof(double);
  double* d_a = nullptr;
  cudaMalloc(&d_a,mem_size);
  double* d_b = nullptr;
  cudaMalloc(&d_b,mem_size);
  double* d_out = nullptr;
  cudaMalloc(&d_out,mem_size);

  cudaMemcpy(d_a, a.data(), mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), mem_size, cudaMemcpyHostToDevice);
  int threadsPerBlock = 256;
  int blocksPerGrid = (vsize + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "CALLING kernel tpb=" << threadsPerBlock << " bpg=" << blocksPerGrid << "\n";
  MyVecAdd<<<blocksPerGrid,threadsPerBlock >>>(d_a,d_b,d_out);
  ARCANE_CHECK_CUDA(cudaDeviceSynchronize());
  ARCANE_CHECK_CUDA(cudaMemcpy(out.data(), d_out, mem_size, cudaMemcpyDeviceToHost));
  for( size_t i=0; i<10; ++i )
    std::cout << "V=" << out[i] << "\n";
  return 0;
}

extern "C"
int arcaneTestCuda2()
{
  MyTestFunc1();
  constexpr int vsize = 2000;
  size_t mem_size = vsize*sizeof(double);
  double* d_a = nullptr;
  cudaMallocManaged(&d_a,mem_size,cudaMemAttachGlobal);
  double* d_b = nullptr;
  cudaMallocManaged(&d_b,mem_size,cudaMemAttachGlobal);
  double* d_out = nullptr;
  cudaMallocManaged(&d_out,mem_size,cudaMemAttachGlobal);

  //d_a = new double[vsize];
  //d_b = new double[vsize];
  //d_out = new double[vsize];

  for( size_t i = 0; i<vsize; ++i ){
    d_a[i] = (double)(i+1);
    d_b[i] = (double)(i*i+1);
    d_out[i] = 0.0; //a[i] + b[i];
  }


  //cudaMemcpy(d_a, a.data(), mem_size, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_b, b.data(), mem_size, cudaMemcpyHostToDevice);
  int threadsPerBlock = 256;
  int blocksPerGrid = (vsize + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "CALLING kernel2 tpb=" << threadsPerBlock << " bpg=" << blocksPerGrid << "\n";
  MyVecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a,d_b,d_out);
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  std::cout << "END OF MYVEC1 e=" << e << " v=" << cudaGetErrorString(e) << "\n";
  //cudaDeviceSynchronize();
  //e = cudaGetLastError();
  //std::cout << "END OF MYVEC2 e=" << e << " v=" << cudaGetErrorString(e) << "\n";
  //cudaMemcpy(out.data(), d_out, mem_size, cudaMemcpyDeviceToHost);
  //e = cudaGetLastError();
  //std::cout << "END OF MYVEC3 e=" << e << " v=" << cudaGetErrorString(e) << "\n";
  for( size_t i=0; i<10; ++i )
    std::cout << "V=" << d_out[i] << "\n";

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C"
int arcaneTestCuda3()
{
  std::cout << "TEST_CUDA_3\n";
  constexpr int vsize = 2000;
  IMemoryAllocator* cuda_allocator = Arcane::Accelerator::Cuda::getCudaMemoryAllocator();
  IMemoryAllocator* cuda_allocator2 = MemoryUtils::getDefaultDataAllocator();
  if (!cuda_allocator2)
    ARCANE_FATAL("platform::getAcceleratorHostMemoryAllocator() is null");
  UniqueArray<double> d_a(cuda_allocator,vsize);
  MyTestFunc1();
  UniqueArray<double> d_b(cuda_allocator,vsize);
  UniqueArray<double> d_out(cuda_allocator,vsize);

  for( size_t i = 0; i<vsize; ++i ){
    d_a[i] = (double)(i+1);
    d_b[i] = (double)(i*i+1);
    d_out[i] = 0.0; //a[i] + b[i];
  }


  int threadsPerBlock = 256;
  int blocksPerGrid = (vsize + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "CALLING kernel2 tpb=" << threadsPerBlock << " bpg=" << blocksPerGrid << "\n";
  MyVecAdd2<<<blocksPerGrid, threadsPerBlock>>>(d_a,d_b,d_out);
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  std::cout << "END OF MYVEC1 e=" << e << " v=" << cudaGetErrorString(e) << "\n";
  for( size_t i=0; i<10; ++i )
    std::cout << "V=" << d_out[i] << "\n";

  // Lance un noyau dynamiquement
  {
    _initArrays(d_a,d_b,d_out,2);

    dim3 dimGrid(threadsPerBlock, 1, 1), dimBlock(blocksPerGrid, 1, 1);

    Span<const double> d_a_span = d_a.span();
    Span<const double> d_b_span = d_b.span();
    Span<double> d_out_view = d_out.span();

    void *kernelArgs[] = {
      (void*)&d_a_span,
      (void*)&d_b_span,
      (void*)&d_out_view
    };
    size_t smemSize = 0;
    cudaStream_t stream;
    ARCANE_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ARCANE_CHECK_CUDA(cudaLaunchKernel((void *)MyVecAdd2, dimGrid, dimBlock, kernelArgs, smemSize, stream));
    ARCANE_CHECK_CUDA(cudaStreamSynchronize(stream));
    for( size_t i=0; i<10; ++i )
      std::cout << "V2=" << d_out[i] << "\n";
  }

  // Lance une lambda
  {
    _initArrays(d_a,d_b,d_out,3);
    Span<const double> d_a_span = d_a.span();
    Span<const double> d_b_span = d_b.span();
    Span<double> d_out_span = d_out.span();
    auto func = [=] ARCCORE_HOST_DEVICE (int i)
                {
                  d_out_span[i] = d_a_span[i] + d_b_span[i];
                  if (i<10){
                    printf("A=%d %lf %lf %lf\n",i,d_a_span[i],d_b_span[i],d_out_span[i]);
                  }};

    MyVecLambda<<<blocksPerGrid,threadsPerBlock>>>(vsize,func);
    ARCANE_CHECK_CUDA(cudaDeviceSynchronize());
    for( size_t i=0; i<10; ++i )
      std::cout << "V3=" << d_out[i] << "\n";

    _initArrays(d_a,d_b,d_out,4);

    // Appelle la version 'hote' de la lambda
    for( int i=0; i<vsize; ++i )
      func(i);
    for( size_t i=0; i<10; ++i )
      std::cout << "V4=" << d_out[i] << "\n";
  }

  // Utilise les Real3
  {
    UniqueArray<Real3> d_a3(cuda_allocator,vsize);
    UniqueArray<Real3> d_b3(cuda_allocator,vsize);
    for( Integer i=0; i<vsize; ++i ){
      Real a = (Real)(i+2);
      Real b = (Real)(i*i+3);

      d_a3[i] = Real3(a,a+1.0,a+2.0);
      d_b3[i] = Real3(b,b+2.0,b+3.0);
    }

    Span<const Real3> d_a3_span = d_a3.span();
    Span<const Real3> d_b3_span = d_b3.span();
    Span<double> d_out_span = d_out.span();
    auto func2 = [=] ARCCORE_HOST_DEVICE (int i) {
                   d_out_span[i] = math::dot(d_a3_span[i],d_b3_span[i]);
                   if (i<10){
                     printf("DOT=%d %lf\n",i,d_out_span[i]);
                   }
                 };
    MyVecLambda<<<blocksPerGrid,threadsPerBlock>>>(vsize,func2);
    ARCANE_CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "TEST WITH REAL3\n";
  }

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C"
int arcaneTestCuda4()
{
  constexpr int vsize = 2000;
  int threadsPerBlock = 256;
  int blocksPerGrid = (vsize + threadsPerBlock - 1) / threadsPerBlock;
  int nb_iteration = 10000;
  Int64 xbegin = platform::getRealTimeNS();
  for(int i=0; i<nb_iteration; ++i )
    MyEmptyKernel<<<blocksPerGrid, threadsPerBlock>>>();
  Int64 xend = platform::getRealTimeNS();
  cudaDeviceSynchronize();
  Int64 xend2 = platform::getRealTimeNS();
  cudaError_t e = cudaGetLastError();
  std::cout << "END OF MyEmptyKernel e=" << e << " v=" << cudaGetErrorString(e) << "\n";
  std::cout << "Time1 = " << (xend-xbegin)/nb_iteration << " Time2=" << (xend2-xbegin)/nb_iteration << "\n";
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C"
int arcaneTestCudaNumArray()
{
  std::cout << "TEST_CUDA_NUM_ARRAY\n";
  constexpr int vsize = 2000;
  IMemoryAllocator* cuda_allocator2 = MemoryUtils::getDefaultDataAllocator();
  if (!cuda_allocator2)
    ARCANE_FATAL("platform::getAcceleratorHostMemoryAllocator() is null");
  NumArray<double,MDDim1> d_a;
  MyTestFunc1();
  NumArray<double,MDDim1> d_b;
  NumArray<double,MDDim1> d_out;
  d_a.resize(vsize);
  d_b.resize(vsize);
  d_out.resize(vsize);
  for( int i = 0; i<vsize; ++i ){
    d_a(i) = (double)(i+1);
    d_b(i) = (double)(i*i+1);
    d_out(i) = 0.0; //a[i] + b[i];
  }


  int threadsPerBlock = 256;
  int blocksPerGrid = (vsize + threadsPerBlock - 1) / threadsPerBlock;
  std::cout << "CALLING kernel2 tpb=" << threadsPerBlock << " bpg=" << blocksPerGrid << "\n";
  MyVecAdd3<<<blocksPerGrid, threadsPerBlock>>>(d_a,d_b,d_out);
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  std::cout << "END OF MYVEC1 e=" << e << " v=" << cudaGetErrorString(e) << "\n";
  for( int i=0; i<10; ++i )
    std::cout << "V=" << d_out(i) << "\n";

  // Lance un noyau dynamiquement
  {
    _initArrays(d_a,d_b,d_out,2);

    dim3 dimGrid(threadsPerBlock, 1, 1), dimBlock(blocksPerGrid, 1, 1);

    MDSpan<const double,MDDim1> d_a_span = d_a.constMDSpan();
    MDSpan<const double,MDDim1> d_b_span = d_b.constMDSpan();
    MDSpan<double,MDDim1> d_out_view = d_out.mdspan();

    void *kernelArgs[] = {
      (void*)&d_a_span,
      (void*)&d_b_span,
      (void*)&d_out_view
    };
    size_t smemSize = 0;
    cudaStream_t stream;
    ARCANE_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ARCANE_CHECK_CUDA(cudaLaunchKernel((void *)MyVecAdd2, dimGrid, dimBlock, kernelArgs, smemSize, stream));
    ARCANE_CHECK_CUDA(cudaStreamSynchronize(stream));
    for( int i=0; i<10; ++i )
      std::cout << "V2=" << d_out(i) << "\n";
  }

  // Lance une lambda
  {
    _initArrays(d_a,d_b,d_out,3);
    MDSpan<const double,MDDim1> d_a_span = d_a.constMDSpan();
    MDSpan<const double,MDDim1> d_b_span = d_b.constMDSpan();
    MDSpan<double,MDDim1> d_out_span = d_out.mdspan();
    auto func = [=] ARCCORE_HOST_DEVICE (int i)
                {
                  d_out_span(i) = d_a_span(i) + d_b_span(i);
                  if (i<10){
                    printf("A=%d %lf %lf %lf\n",i,d_a_span(i),d_b_span(i),d_out_span(i));
                  }};

    MyVecLambda<<<blocksPerGrid,threadsPerBlock>>>(vsize,func);
    ARCANE_CHECK_CUDA(cudaDeviceSynchronize());
    for( int i=0; i<10; ++i )
      std::cout << "V3=" << d_out(i) << "\n";

    _initArrays(d_a,d_b,d_out,4);

    // Appelle la version 'hote' de la lambda
    for( int i=0; i<vsize; ++i )
      func(i);
    for( int i=0; i<10; ++i )
      std::cout << "V4=" << d_out(i) << "\n";
  }

  // Utilise les Real3 avec un tableau multi-dimensionel
  {
    NumArray<Real,MDDim2> d_a3(vsize,3);
    NumArray<Real,MDDim2> d_b3(vsize,3);
    for( Integer i=0; i<vsize; ++i ){
      Real a = (Real)(i+2);
      Real b = (Real)(i*i+3);

      d_a3(i,0) = a;
      d_a3(i,1) = a + 1.0;
      d_a3(i,2) = a + 2.0;

      d_b3(i,0) = b;
      d_b3(i,1) = b + 1.0;
      d_b3(i,2) = b + 2.0;
    }

    MDSpan<const Real,MDDim2> d_a3_span = d_a3.constMDSpan();
    MDSpan<const Real,MDDim2> d_b3_span = d_b3.constMDSpan();
    MDSpan<double,MDDim1> d_out_span = d_out.mdspan();
    auto func2 = [=] ARCCORE_HOST_DEVICE (int i) {
                   Real3 xa(d_a3_span(i,0),d_a3_span(i,1),d_a3_span(i,2));
                   Real3 xb(d_b3_span(i,0),d_b3_span(i,1),d_b3_span(i,2));
                   d_out_span(i) = math::dot(xa,xb);
                   if (i<10){
                     printf("DOT NUMARRAY=%d %lf\n",i,d_out_span(i));
                   }
                 };
    MyVecLambda<<<blocksPerGrid,threadsPerBlock>>>(vsize,func2);
    ARCANE_CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "TEST WITH REAL3\n";
  }

  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/Reduce.h"

namespace ax = Arcane::Accelerator;

void arcaneTestCudaReductionX(int vsize,ax::RunQueue& queue)
{
  using namespace Arcane::Accelerator;
  std::cout << "TestReduction vsize=" << vsize << " accelerator_policy=" << queue.executionPolicy() << "\n";
  NumArray<int,MDDim1> d_a(vsize);
  NumArray<int,MDDim1> d_out(vsize);
  for( Integer i=0; i<vsize; ++i ){
    int a = 5 + ((i+2) % 43);
    d_a[i] = a;
    d_out[i] = 0;
    //std::cout << "I=" << i << " a=" << a << "\n";
  }
  RunCommand command = makeCommand(queue);
  ReducerSum<int> sum_reducer(command);
  ReducerSum<double> sum_double_reducer(command);
  ReducerMax<int> max_int_reducer(command);
  ReducerMax<double> max_double_reducer(command);
  ReducerMin<int> min_int_reducer(command);
  ReducerMin<double> min_double_reducer(command);
  Span<const int> xa = d_a.to1DSpan();
  Span<int> xout = d_out.to1DSpan();
  command << RUNCOMMAND_LOOP1(idx,vsize)
  {
    auto [i] = idx();
    double vxa = (double)(xa[i]);
    xout[i] = xa[i];
    sum_reducer.add(xa[i]);
    sum_double_reducer.add(vxa);
    max_int_reducer.max(xa[i]);
    max_double_reducer.max(vxa);
    min_int_reducer.min(xa[i]);
    min_double_reducer.min(vxa);
    //if (i<10)
    //printf("Do Reduce i=%d v=%d %lf\n",i,xa[i],vxa);
  };
  //run(command,ArrayBounds<MDDim1>(vsize),func2);
  std::cout << "SumReducer v_int=" << sum_reducer.reduce()
            << " v_double=" << sum_double_reducer.reduce()
            << "\n";
  std::cout << "MaxReducer v_int=" << max_int_reducer.reduce()
            << " v_double=" << max_double_reducer.reduce()
            << "\n";
  std::cout << "MinReducer v_int=" << min_int_reducer.reduce()
            << " v_double=" << min_double_reducer.reduce()
            << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C"
int arcaneTestCudaReduction()
{
  // TODO: tester en ne commancant pas par 0.
  ax::Runner runner_seq(ax::eExecutionPolicy::Sequential);
  ax::Runner runner_thread(ax::eExecutionPolicy::Thread);
  ax::Runner runner_cuda(ax::eExecutionPolicy::CUDA);
  ax::RunQueue queue1{makeQueue(runner_seq)};
  ax::RunQueue queue2{makeQueue(runner_thread)};
  ax::RunQueue queue3{makeQueue(runner_cuda)};
  int sizes_to_test[] = { 56, 567, 452182 };
  for( int i=0; i<3; ++i ){
    int vsize = sizes_to_test[i];
    arcaneTestCudaReductionX(vsize,queue1);
    arcaneTestCudaReductionX(vsize,queue2);
    arcaneTestCudaReductionX(vsize,queue3);
  }
  return 0;
}
