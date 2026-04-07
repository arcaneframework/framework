// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/utils/ObjectWithTrace.h>
#include "HypreInternal.h"
#include <numeric>

#ifdef ALIEN_USE_CUDA
#include <cuda_runtime.h>
#endif
#ifdef ALIEN_USE_HIP
#include <hip_runtime.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Internal {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixInternal::~MatrixInternal()
{
  if (m_internal)
    HYPRE_IJMatrixDestroy(m_internal);
}


/*---------------------------------------------------------------------------*/

bool
MatrixInternal::init(const HYPRE_Int ilower, const HYPRE_Int iupper, const HYPRE_Int jlower,
    const int jupper, const Arccore::ConstArrayView<Arccore::Integer>& lineSizes)
{
  HYPRE_ClearAllErrors() ;

  int ierr = 0; // code d'erreur de retour

  // -- Matrix --
  ierr = HYPRE_IJMatrixCreate(m_comm, ilower, iupper, jlower, jupper, &m_internal);
  ierr |= HYPRE_IJMatrixSetObjectType(m_internal, HYPRE_PARCSR);
  if(lineSizes.size()>0)
     ierr |= HYPRE_IJMatrixSetRowSizes(m_internal, lineSizes.unguardedBasePointer());
  ierr |= HYPRE_IJMatrixInitialize(m_internal);
  return (ierr == 0 || ierr == HYPRE_ERROR_CONV);
}

bool
VectorInternal::init(const int ilower, const int iupper)
{
  // -- B Vector --
  int ierr = HYPRE_IJVectorCreate(m_comm, ilower, iupper, &m_internal);
  ierr |= HYPRE_IJVectorSetObjectType(m_internal, HYPRE_PARCSR);
  ierr |= HYPRE_IJVectorInitialize(m_internal);

  return (ierr == 0 || ierr == HYPRE_ERROR_CONV);
}

/*---------------------------------------------------------------------------*/

bool
MatrixInternal::addMatrixValues(const int nrow, const int* rows, const int* ncols,
    const int* cols, const Arccore::Real* values)
{
  int ierr = HYPRE_IJMatrixAddToValues(
      m_internal, nrow, const_cast<int*>(ncols), rows, cols, values);
  return (ierr == 0 || ierr == HYPRE_ERROR_CONV);
}

/*---------------------------------------------------------------------------*/

bool
MatrixInternal::setMatrixValues(const int nrow, const int* rows, const int* ncols,
    const int* cols, const Arccore::Real* values)
{
#ifdef PRINT_DEBUG_INFO
  alien_info([&] {
  cout()<<"HYPREMATRXIXINTERNAL SET MATRXI VALUES : "<<nrow<<" "<<m_memory_type;
  });

  if(m_memory_type==BackEnd::Memory::Host)
  {
    int nnz = std::accumulate(ncols,ncols+nrow,0);
    alien_info([&]{
      cout()<<"MATRIX : "<<nrow<<" "<<nnz;
    int offset = 0 ;
    for(int i=0;i<nrow;++i)
    {
      int row_size = ncols[i];
      cout()<<"MAT["<<i<<","<<row_size<<"]:";
      for(int k=0;k<row_size;++k)
      {
        cout()<<'\t'<<cols[offset+k]<<" "<<values[offset+k]<<" ";
      }
      offset += row_size;
    }
    });
  }
  else
  {
    std::vector<int> row_size(nrow) ;
    cudaMemcpy(row_size.data(), ncols, nrow * sizeof(int), cudaMemcpyDeviceToHost);
    int nnz = std::accumulate(row_size.begin(),row_size.end(),0);
    std::vector<ValueType> val(nnz) ;
    cudaMemcpy(val.data(), values, nnz * sizeof(ValueType), cudaMemcpyDeviceToHost);
    std::vector<int> h_cols(nnz) ;
    cudaMemcpy(h_cols.data(), cols, nnz * sizeof(int), cudaMemcpyDeviceToHost);

    alien_info([&]{
      cout()<<"MATRIX : "<<nrow<<" "<<nnz;
    int offset = 0 ;
    for(int i=0;i<nrow;++i)
    {
      cout()<<"MAT["<<i<<","<<row_size[i]<<"]:";
      for(int k=0;k<row_size[i];++k)
      {
        cout()<<'\t'<<h_cols[offset+k]<<" "<<val[offset+k]<<" ";
      }
      offset += row_size[i];
    }
    });
  }
#endif
  int ierr = HYPRE_IJMatrixSetValues(
      m_internal, nrow, const_cast<int*>(ncols), rows, cols, values);
  return (ierr == 0 || ierr == HYPRE_ERROR_CONV);
}

void
MatrixInternal::allocateHostPointers(std::size_t nrows,
                                     std::size_t nnz,
                                     IndexType** rows,
                                     IndexType** ncols,
                                     IndexType** cols,
                                     ValueType** values)
{
#ifdef ALIEN_USE_CUDA
  cudaMallocHost(rows, nrows * sizeof(HYPRE_BigInt));
  cudaMallocHost(ncols, nrows * sizeof(HYPRE_Int));
  cudaMallocHost(cols, nnz * sizeof(HYPRE_BigInt));
  cudaMallocHost(values, nnz * sizeof(ValueType));
#endif
#ifdef ALIEN_USE_HIP
  hipHostMalloc(rows, nrows * sizeof(HYPRE_BigInt));
  hipHostMalloc(ncols, nrows * sizeof(HYPRE_Int));
  hipHostMalloc(cols, nnz * sizeof(HYPRE_BigInt));
  hipHostMalloc(values, nnz * sizeof(ValueType));
#endif

}

void
MatrixInternal::freeHostPointers(IndexType* rows,
                                 IndexType* ncols,
                                 IndexType* cols,
                                 ValueType* values)
{
#ifdef ALIEN_USE_CUDA
  cudaFreeHost(rows);
  cudaFreeHost(ncols);
  cudaFreeHost(cols);
  cudaFreeHost(values);
#endif
#ifdef ALIEN_USE_HIP
  hipHostFree(rows);
  hipHostFree(ncols);
  hipHostFree(cols);
  hipHostFree(values);
#endif

}

void
MatrixInternal::allocateDevicePointers(std::size_t nrows,
                                       std::size_t nnz,
                                       IndexType** rows,
                                       IndexType** ncols,
                                       IndexType** cols,
                                       ValueType** values)
{
#ifdef ALIEN_USE_CUDA
  cudaMalloc(rows, nrows * sizeof(IndexType));
  cudaMalloc(ncols, nrows * sizeof(IndexType));
  cudaMalloc(cols, nnz * sizeof(IndexType));
  cudaMalloc(values, nnz * sizeof(ValueType));
#endif
#ifdef ALIEN_USE_HIP
  hipMalloc(rows, nrows * sizeof(IndexType));
  hipMalloc(ncols, nrows * sizeof(IndexType));
  hipMalloc(cols, nnz * sizeof(IndexType));
  hipMalloc(values, nnz * sizeof(ValueType));
#endif
}

void
MatrixInternal::freeDevicePointers(IndexType* rows,
                                   IndexType* ncols,
                                   IndexType* cols,
                                   ValueType* values)
{
#ifdef ALIEN_USE_CUDA
  cudaFree(rows);
  cudaFree(ncols);
  cudaFree(cols);
  cudaFree(values);
#endif
#ifdef ALIEN_USE_HIP
  hipFree(rows);
  hipFree(ncols);
  hipFree(cols);
  hipFree(values);
#endif

}

;

 void
 MatrixInternal::copyHostToDevicePointers(std::size_t nrows,
                                          std::size_t nnz,
                                          const IndexType* rows_h,
                                          const IndexType* ncols_h,
                                          const IndexType* cols_h,
                                          const ValueType* values_h,
                                          IndexType* rows_d,
                                          IndexType* ncols_d,
                                          IndexType* cols_d,
                                          ValueType* values_d)
{
#ifdef ALIEN_USE_CUDA
  // Copier Host -> Device
  cudaMemcpy(rows_d, rows_h, nrows * sizeof(HYPRE_BigInt), cudaMemcpyHostToDevice);
  cudaMemcpy(ncols_d, ncols_h, nrows * sizeof(HYPRE_Int), cudaMemcpyHostToDevice);
  cudaMemcpy(cols_d, cols_h, nnz * sizeof(HYPRE_BigInt), cudaMemcpyHostToDevice);
  cudaMemcpy(values_d, values_h, nnz * sizeof(HYPRE_Complex), cudaMemcpyHostToDevice);
#endif
#ifdef ALIEN_USE_HIP
  // Copier Host -> Device
  hipMemcpy(rows_d, rows_h, nrows * sizeof(HYPRE_BigInt), hipMemcpyHostToDevice);
  hipMemcpy(ncols_d, ncols_h, nrows * sizeof(HYPRE_Int), hipMemcpyHostToDevice);
  hipMemcpy(cols_d, cols_h, nnz * sizeof(HYPRE_BigInt), hipMemcpyHostToDevice);
  hipMemcpy(values_d, values_h, nnz * sizeof(HYPRE_Complex), hipMemcpyHostToDevice);
#endif

}

void
MatrixInternal::copyDeviceToHostPointers(std::size_t nrows,
                                         std::size_t nnz,
                                         const IndexType* rows_d,
                                         const IndexType* ncols_d,
                                         const IndexType* cols_d,
                                         const ValueType* values_d,
                                         IndexType* rows_h,
                                         IndexType* ncols_h,
                                         IndexType* cols_h,
                                         ValueType* values_h)
{
#ifdef ALIEN_USE_CUDA
 // Copier Device -> Host
 cudaMemcpy(rows_h, rows_d    , nrows * sizeof(HYPRE_BigInt),  cudaMemcpyDeviceToHost);
 cudaMemcpy(ncols_h, ncols_d  , nrows * sizeof(HYPRE_Int),     cudaMemcpyDeviceToHost);
 cudaMemcpy(cols_h, cols_d    , nnz   * sizeof(HYPRE_BigInt),  cudaMemcpyDeviceToHost);
 cudaMemcpy(values_h, values_d, nnz   * sizeof(HYPRE_Complex), cudaMemcpyDeviceToHost);
#endif
#ifdef ALIEN_USE_HIP
 // Copier Device -> Host
 hipMemcpy(rows_h, rows_d    , nrows * sizeof(HYPRE_BigInt),  hipMemcpyDeviceToHost);
 hipMemcpy(ncols_h, ncols_d  , nrows * sizeof(HYPRE_Int),     hipMemcpyDeviceToHost);
 hipMemcpy(cols_h, cols_d    , nnz   * sizeof(HYPRE_BigInt),  hipMemcpyDeviceToHost);
 hipMemcpy(values_h, values_d, nnz   * sizeof(HYPRE_Complex), hipMemcpyDeviceToHost);
#endif

}
/*---------------------------------------------------------------------------*/
VectorInternal::~VectorInternal()
{
  if (m_internal)
    HYPRE_IJVectorDestroy(m_internal);

  if(m_memory_type==BackEnd::Memory::Device)
  {
#ifdef ALIEN_USE_CUDA
    if(m_rows)
      cudaFree(m_rows);
    if(m_zeros_device)
      cudaFree(m_zeros_device);
#endif
#ifdef ALIEN_USE_HIP
    if(m_rows)
      hipFree(m_rows);
    if(m_zeros_device)
      hipFree(m_zeros_device);
#endif

  }
}

void VectorInternal::setRows(std::size_t nrow,IndexType const* h_rows)
{
  if(m_memory_type==BackEnd::Memory::Device)
  {
#ifdef ALIEN_USE_CUDA
    cudaMalloc(&m_rows, nrow * sizeof(IndexType));
    cudaMemcpy(m_rows, h_rows, nrow * sizeof(HYPRE_BigInt), cudaMemcpyHostToDevice);
#endif
#ifdef ALIEN_USE_HIP
    hipMalloc(&m_rows, nrow * sizeof(IndexType));
    hipMemcpy(m_rows, h_rows, nrow * sizeof(HYPRE_BigInt), hipMemcpyHostToDevice);
#endif
  }
}

bool
VectorInternal::addValues(const int nrow, const int* rows, const Arccore::Real* values)
{
  int ierr = HYPRE_IJVectorAddToValues(m_internal,
      nrow, // nb de valeurs
      rows, values);
  return (ierr == 0 || ierr == HYPRE_ERROR_CONV);
}

/*---------------------------------------------------------------------------*/

bool
VectorInternal::setValues(const int nrow, const int* rows, const Arccore::Real* values)
{
#ifdef PRINT_DEBUG_INFO
  if(m_memory_type==BackEnd::Memory::Host)
  {
    for(int i=0;i<nrow;++i)
      std::cout<<"HYPRE B["<<i<<"]="<<values[i]<<std::endl ;
  }
  else
  {
    std::vector<ValueType> val(nrow) ;
       cudaMemcpy(val.data(), values, nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);
       for(int i=0;i<nrow;++i)
         std::cout<<"HYPRE B["<<i<<"]="<<val[i]<<std::endl ;
  }
#endif
  int ierr = HYPRE_IJVectorSetValues(m_internal,
      nrow, // nb de valeurs
      rows, values);
  return (ierr == 0 || ierr == HYPRE_ERROR_CONV);
}

bool
VectorInternal::setValuesOnDevice(const int nrow,  const int* rows, const Arccore::Real* values)
{
  if(m_rows==nullptr)
    setRows(nrow,rows) ;
  /*
  {
    std::vector<ValueType> val(nrow) ;
    cudaMemcpy(val.data(), values, nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);
    for(int i=0;i<nrow;++i)
      std::cout<<"HYPRE B["<<i<<"]="<<val[i]<<std::endl ;
  }*/
  int ierr = HYPRE_IJVectorSetValues(m_internal,
      nrow, // nb de valeurs
      m_rows, values);
  return (ierr == 0 || ierr == HYPRE_ERROR_CONV);
}

/*---------------------------------------------------------------------------*/

bool
VectorInternal::setInitValues(
    const int nrow, const int* rows, const Arccore::Real* values)
{
  int ierr = HYPRE_IJVectorSetValues(m_internal,
      nrow, // nb de valeurs
      rows, values);
  return (ierr == 0 || ierr == HYPRE_ERROR_CONV);
}

bool
VectorInternal::setValuesToZeros(const int nrow,  const int* rows)
{
  HYPRE_ParVector par_vector;
  HYPRE_IJVectorGetObject(m_internal, reinterpret_cast<void **>(&par_vector));
  int ierr = HYPRE_ParVectorSetConstantValues(par_vector, 0.0);
  return (ierr == 0 || ierr == HYPRE_ERROR_CONV);
  /*
  if(m_memory_type==BackEnd::Memory::Host)
  {
    if(m_zeros.size()==0)
      m_zeros.resize(nrow,0.) ;
    return setValues(nrow,rows,m_zeros.data()) ;
  }
  else
  {
    if(m_rows==nullptr)
      setRows(nrow,rows) ;
    if(m_zeros_device==nullptr)
    {
#ifdef ALIEN_USE_CUDA
      cudaMalloc(&m_zeros_device, nrow * sizeof(ValueType));
      cudaMemset(m_zeros_device,0,nrow * sizeof(ValueType)) ;
#endif
    }
    return setValues(nrow,m_rows,m_zeros_device) ;
  }*/
}
/*---------------------------------------------------------------------------*/

bool
MatrixInternal::assemble()
{
  int ierr = HYPRE_IJMatrixAssemble(m_internal);
  return (ierr == 0);
}

bool
VectorInternal::assemble()
{
  int ierr = HYPRE_IJVectorAssemble(m_internal);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
VectorInternal::getValues(const int nrow, const int* rows, Arccore::Real* values)
{
  int ierr = HYPRE_IJVectorGetValues(m_internal, nrow, rows, values);

#ifdef PRINT_DEBUG_INFO
  if(m_memory_type==BackEnd::Memory::Host)
  {
    for(int i=0;i<nrow;++i)
      std::cout<<"HYPRE X["<<i<<"]="<<values[i]<<std::endl ;
  }
#endif
  return (ierr == 0);
}


bool
VectorInternal::getValuesToDevice(const int nrow, const int* rows, Arccore::Real* values_d)
{
  if(m_rows==nullptr)
    setRows(nrow,rows) ;

  if(m_memory_type==BackEnd::Memory::Host)
  {
    UniqueArray<Arccore::Real> values(nrow) ;
    int ierr = HYPRE_IJVectorGetValues(m_internal, nrow, rows, values.data());
#ifdef ALIEN_USE_CUDA
    cudaMemcpy(values_d, values.data(), nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);
#endif
#ifdef ALIEN_USE_HIP
    hipMemcpy(values_d, values.data(), nrow * sizeof(ValueType), hipMemcpyDeviceToHost);
#endif
    return (ierr == 0);
  }
  else
  {
    int ierr = HYPRE_IJVectorGetValues(m_internal, nrow, m_rows, values_d);

#ifdef PRINT_DEBUG_INFO
    {
      std::vector<ValueType> val(nrow) ;
      cudaMemcpy(val.data(), values_d, nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);
      for(int i=0;i<nrow;++i)
        std::cout<<"HYPRE X["<<i<<"]="<<val[i]<<std::endl ;
    }
#endif
    return (ierr == 0);
  }
}

bool
VectorInternal::getValuesToHost(const int nrow, const int* rows, Arccore::Real* values_h)
{
  if(m_rows==nullptr)
    setRows(nrow,rows) ;

  if(m_memory_type==BackEnd::Memory::Host)
  {
    int ierr = HYPRE_IJVectorGetValues(m_internal, nrow, rows, values_h);
    return (ierr == 0);
  }
  else
  {
#ifdef ALIEN_USE_CUDA
    if(m_rows==nullptr)
      setRows(nrow,rows) ;
    ValueType* values_d ;
    cudaMalloc(&values_d, nrow * sizeof(ValueType));
    int ierr = HYPRE_IJVectorGetValues(m_internal, nrow, m_rows, values_d);
    cudaMemcpy(values_h, values_d, nrow * sizeof(ValueType), cudaMemcpyDeviceToHost);
    cudaFree(values_d);
    return (ierr == 0);
#else
#ifdef ALIEN_USE_HIP
    if(m_rows==nullptr)
      setRows(nrow,rows) ;
    ValueType* values_d;
    hipMalloc(&values_d, nrow * sizeof(ValueType));
    int ierr = HYPRE_IJVectorGetValues(m_internal, nrow, m_rows, values_d);
    hipMemcpy(values_h, values_d, nrow * sizeof(ValueType), hipMemcpyDeviceToHost);
    hipFree(values_d);
    return (ierr == 0);
#else
    return false ;
#endif
#endif
  }
}
void VectorInternal::allocateDevicePointers(std::size_t local_size, ValueType** values)
{
#ifdef ALIEN_USE_CUDA
      cudaMalloc(values, local_size * sizeof(ValueType));
#endif
#ifdef ALIEN_USE_HIP
      hipMalloc(values, local_size * sizeof(ValueType));
#endif

}


void VectorInternal::freeDevicePointers(ValueType* values)
{
#ifdef ALIEN_USE_CUDA
  cudaFree(values);
#endif
#ifdef ALIEN_USE_HIP
  hipFree(values);
#endif

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
