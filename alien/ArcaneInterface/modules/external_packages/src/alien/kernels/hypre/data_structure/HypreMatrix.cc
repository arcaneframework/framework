// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <alien/kernels/hypre/linear_solver/HypreInternalLinearSolver.h>
#include "HypreMatrix.h"
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
    return m_internal->setMatrixValuesFrom(nrow, nnz, rows, ncols, cols, values, memory);
  }
}
/*---------------------------------------------------------------------------*/

bool
HypreMatrix::assemble()
{
  return m_internal->assemble();
}

HypreMatrix::CSRView::CSRView(HypreMatrix const* parent,
                              BackEnd::Memory::eType memory,
                              int nrows,
                              int nnz)
: m_parent(parent)
, m_memory(memory)
, m_nrows(nrows)
, m_nnz(nnz)
{
  switch(m_memory)
  {
    case BackEnd::Memory::Device :
      m_parent->m_internal->initDevicePointer(m_nrows,m_nnz,&m_rows,&m_ncols,&m_cols,&m_values) ;
    break ;
    case BackEnd::Memory::Host :
    default:
      m_parent->m_internal->initHostPointer(m_nrows,m_nnz,&m_rows,&m_ncols,&m_cols,&m_values) ;
    break ;

  }
}
HypreMatrix::CSRView::~CSRView()
{
  switch(m_memory)
  {
    case BackEnd::Memory::Host :
      m_parent->m_internal->freeHostPointer(m_rows,m_ncols,m_cols,m_values) ;
    break ;
    case BackEnd::Memory::Device :
      m_parent->m_internal->freeDevicePointer(m_rows,m_ncols,m_cols,m_values) ;
    break ;
  }
}
HypreMatrix::CSRView
HypreMatrix::csrView(BackEnd::Memory::eType memory, int nrows, int nnz)
{
  return CSRView(this,memory, nrows, nnz) ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
