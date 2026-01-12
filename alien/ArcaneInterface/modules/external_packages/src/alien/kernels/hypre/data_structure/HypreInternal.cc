// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "HypreInternal.h"

#ifdef ALIEN_USE_CUDA
#include <cuda_runtime.h>
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

VectorInternal::~VectorInternal()
{
  if (m_internal)
    HYPRE_IJVectorDestroy(m_internal);
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
  return (ierr == 0);
}

bool
VectorInternal::init(const int ilower, const int iupper)
{
  // -- B Vector --
  int ierr = HYPRE_IJVectorCreate(m_comm, ilower, iupper, &m_internal);
  ierr |= HYPRE_IJVectorSetObjectType(m_internal, HYPRE_PARCSR);
  ierr |= HYPRE_IJVectorInitialize(m_internal);

  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
MatrixInternal::addMatrixValues(const int nrow, const int* rows, const int* ncols,
    const int* cols, const Arccore::Real* values)
{
  int ierr = HYPRE_IJMatrixAddToValues(
      m_internal, nrow, const_cast<int*>(ncols), rows, cols, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
MatrixInternal::setMatrixValues(const int nrow, const int* rows, const int* ncols,
    const int* cols, const Arccore::Real* values)
{
  int ierr = HYPRE_IJMatrixSetValues(
      m_internal, nrow, const_cast<int*>(ncols), rows, cols, values);
  return (ierr == 0);
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
}

/*---------------------------------------------------------------------------*/

bool
VectorInternal::addValues(const int nrow, const int* rows, const Arccore::Real* values)
{
  int ierr = HYPRE_IJVectorAddToValues(m_internal,
      nrow, // nb de valeurs
      rows, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
VectorInternal::setValues(const int nrow, const int* rows, const Arccore::Real* values)
{
  int ierr = HYPRE_IJVectorSetValues(m_internal,
      nrow, // nb de valeurs
      rows, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
VectorInternal::setInitValues(
    const int nrow, const int* rows, const Arccore::Real* values)
{
  int ierr = HYPRE_IJVectorSetValues(m_internal,
      nrow, // nb de valeurs
      rows, values);
  return (ierr == 0);
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
  int ierr;
  ierr = HYPRE_IJVectorGetValues(m_internal, nrow, rows, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
