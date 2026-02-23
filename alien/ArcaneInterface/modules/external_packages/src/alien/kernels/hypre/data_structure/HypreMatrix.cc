// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/kernels/hypre/linear_solver/HypreInternalLinearSolver.h>

#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/data_structure/HypreVector.h>
#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreInternal.h>
#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/data/ISpace.h>

#include <arccore/message_passing/Communicator.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HypreMatrix::HypreMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::hypre>::name())
, m_internal(nullptr)
, m_pm(nullptr)
{
  const auto& row_space = multi_impl->rowSpace();
  const auto& col_space = multi_impl->colSpace();
  if (row_space.size() != col_space.size())
    throw Arccore::FatalErrorException("Hypre matrix must be square");
  m_pm = multi_impl->distribution().parallelMng();
}

/*---------------------------------------------------------------------------*/

HypreMatrix::~HypreMatrix()
{
  delete m_internal;
}

BackEnd::Memory::eType HypreMatrix::getMemoryType() const {
  return m_internal->getMemoryType() ;
}

BackEnd::Exec::eSpaceType HypreMatrix::getExecSpace() const {
  return m_internal->getExecSpace() ;
}

/*---------------------------------------------------------------------------*/

bool
HypreMatrix::initMatrix(const int ilower, const int iupper, const int jlower,
    const int jupper, const Arccore::ConstArrayView<Arccore::Integer>& lineSizes)
{
  delete m_internal;
  auto memory_type = HypreInternalLinearSolver::m_library_plugin->getMemoryType() ;
  auto exec_space = HypreInternalLinearSolver::m_library_plugin->getExecSpace() ;
  auto pm = m_pm->communicator();
  if (pm.isValid()) {
    m_internal = new MatrixInternal(static_cast<MPI_Comm>(pm), memory_type, exec_space);
  }
  else {
    m_internal = new MatrixInternal(MPI_COMM_WORLD,
        memory_type,
        exec_space);
  }
  return m_internal->init(ilower, iupper, jlower, jupper, lineSizes);
}

/*---------------------------------------------------------------------------*/

bool
HypreMatrix::addMatrixValues(const int nrow, const int* rows, const int* ncols,
    const int* cols, const Arccore::Real* values)
{
  return m_internal->addMatrixValues(nrow, rows, ncols, cols, values);
}

/*---------------------------------------------------------------------------*/

bool
HypreMatrix::setMatrixValues(const int nrow, const int* rows, const int* ncols,
    const int* cols, const Arccore::Real* values)
{
  return m_internal->setMatrixValues(nrow, rows, ncols, cols, values);
}

bool
HypreMatrix::setMatrixValuesFrom(const int nrow,
                                 const int nnz,
                                 const int* rows,
                                 const int* ncols,
                                 const int* cols,
                                 const Arccore::Real* values,
                                 BackEnd::Memory::eType memory)
{
  if(memory == m_internal->getMemoryType())
    return m_internal->setMatrixValues(nrow, rows, ncols, cols, values);
  else
  {
#define NEED_COPY_DATA
#ifdef NEED_COPY_DATA
    auto view = HCSRView{this,m_internal->getMemoryType(),std::size_t(nrow),std::size_t(nnz)} ;
    if(memory == BackEnd::Memory::Host)
      m_internal->copyHostToDevicePointers(nrow, nnz,
                                           rows, ncols, cols, values,
                                           view.m_rows, view.m_ncols, view.m_cols, view.m_values) ;
    else
      m_internal->copyDeviceToHostPointers(nrow, nnz,
                                           rows, ncols, cols, values,
                                           view.m_rows, view.m_ncols, view.m_cols, view.m_values) ;

     return m_internal->setMatrixValues(nrow, view.m_rows, view.m_ncols, view.m_cols, view.m_values);
#else
     return m_internal->setMatrixValues(nrow, rows, ncols, cols, values);
#endif
  }
}
/*---------------------------------------------------------------------------*/

bool
HypreMatrix::assemble()
{
  return m_internal->assemble();
}

void HypreMatrix::allocateDevicePointers(std::size_t nrows,
                                         std::size_t nnz,
                                         HypreMatrix::IndexType** rows,
                                         HypreMatrix::IndexType** ncols,
                                         HypreMatrix::IndexType** cols,
                                         HypreMatrix::ValueType** values) const
{
  m_internal->allocateDevicePointers(nrows, nnz, rows, ncols, cols, values) ;
}

void HypreMatrix::freeDevicePointers(HypreMatrix::IndexType* rows,
                                     HypreMatrix::IndexType* ncols,
                                     HypreMatrix::IndexType* cols,
                                     HypreMatrix::ValueType* values) const
{
  m_internal->freeDevicePointers(rows, ncols, cols, values) ;
}

void HypreMatrix::allocateHostPointers(std::size_t nrows,
                                       std::size_t nnz,
                                       HypreMatrix::IndexType** rows,
                                       HypreMatrix::IndexType** ncols,
                                       HypreMatrix::IndexType** cols,
                                       HypreMatrix::ValueType** values) const
{
  m_internal->allocateHostPointers(nrows, nnz, rows, ncols, cols, values) ;
}

void HypreMatrix::freeHostPointers(HypreMatrix::IndexType* rows,
                                     HypreMatrix::IndexType* ncols,
                                     HypreMatrix::IndexType* cols,
                                     HypreMatrix::ValueType* values) const
{
  m_internal->freeHostPointers(rows, ncols, cols, values) ;
}

void HypreMatrix::copyHostToDevicePointers(std::size_t nrows,
                              std::size_t nnz,
                              const IndexType* rows_h,
                              const IndexType* ncols_h,
                              const IndexType* cols_h,
                              const ValueType* values_h,
                              IndexType* rows_d,
                              IndexType* ncols_d,
                              IndexType* cols_d,
                              ValueType* values_d) const
{
  m_internal->copyHostToDevicePointers(nrows,
                                       nnz,
                                       rows_h,
                                       ncols_h,
                                       cols_h,
                                       values_h,
                                       rows_d,
                                       ncols_d,
                                       cols_d,
                                       values_d);
}

void HypreMatrix::copyDeviceToHostPointers(std::size_t nrows,
                              std::size_t nnz,
                              const IndexType* rows_h,
                              const IndexType* ncols_h,
                              const IndexType* cols_h,
                              const ValueType* values_h,
                              IndexType* rows_d,
                              IndexType* ncols_d,
                              IndexType* cols_d,
                              ValueType* values_d) const
{
  m_internal->copyDeviceToHostPointers(nrows,
                                       nnz,
                                       rows_h,
                                       ncols_h,
                                       cols_h,
                                       values_h,
                                       rows_d,
                                       ncols_d,
                                       cols_d,
                                       values_d);
}

HypreMatrix::HCSRView
HypreMatrix::hcsrView(BackEnd::Memory::eType memory, int nrows, int nnz)
{
  return HCSRView(this,memory, nrows, nnz) ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
