// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <alien/kernels/hypre/linear_solver/HypreInternalLinearSolver.h>
#include "HypreVector.h"

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/sycl/data/SYCLVector.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreInternal.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
#include <arccore/message_passing/Communicator.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HypreVector::HypreVector(const MultiVectorImpl* multi_impl)
: IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::hypre>::name())
, m_internal(nullptr)
, m_block_size(1)
, m_offset(0)
{
  ;
}

/*---------------------------------------------------------------------------*/

HypreVector::~HypreVector()
{
  delete m_internal;
}


BackEnd::Memory::eType HypreVector::getMemoryType() const {
  return m_internal->getMemoryType() ;
}

BackEnd::Exec::eSpaceType HypreVector::getExecSpace() const {
  return m_internal->getExecSpace() ;
}
/*---------------------------------------------------------------------------*/

void
HypreVector::init(const VectorDistribution& dist, const bool need_allocate)
{
  const Block* block = this->block();
  if (this->block())
    m_block_size *= block->size();
  else if (this->vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Not implemented yet");
  else
    m_block_size = 1;
  m_offset = dist.offset();
  if (need_allocate)
  {
    allocate(dist);
  }
}

void
HypreVector::init(const VectorDistribution& dist, Integer block_size, const bool need_allocate)
{
  const Block* block = this->block();
  if (this->block())
    m_block_size *= block->size();
  else if (this->vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Not implemented yet");
  else
    m_block_size = block_size;
  m_offset = dist.offset();
  if (need_allocate)
  {
    allocate(dist);
  }
}

/*---------------------------------------------------------------------------*/

void
HypreVector::allocate(const VectorDistribution& dist)
{
  delete m_internal;
  auto memory_type = HypreInternalLinearSolver::m_library_plugin->getMemoryType() ;
  auto exec_space = HypreInternalLinearSolver::m_library_plugin->getExecSpace() ;
  //const VectorDistribution& dist = this->distribution();
  auto pm = dist.parallelMng()->communicator();
  if (pm.isValid()) {
    m_internal =
        new VectorInternal(static_cast<const MPI_Comm>(pm), memory_type, exec_space);
  }
  else {
    m_internal = new VectorInternal(MPI_COMM_WORLD,
        memory_type,
        exec_space);
  }
  int ilower = dist.offset() * m_block_size;
  int iupper = ilower + dist.localSize() * m_block_size - 1;
  m_internal->init(ilower, iupper);
  m_rows.resize(dist.localSize() * m_block_size);
  for (int i = 0; i < dist.localSize() * m_block_size; ++i)
    m_rows[i] = ilower + i;
  m_internal->setRows(m_rows.size(),m_rows.data()) ;
}

/*---------------------------------------------------------------------------*/

bool
HypreVector::setValues(const int nrow, const int* rows, const double* values)
{
  if (m_internal == nullptr)
    return false;
  return m_internal->setValues(nrow, rows, values);
}

/*---------------------------------------------------------------------------*/

bool
HypreVector::setValues(const int nrow, const double* values)
{
  if (m_internal == nullptr)
    return false;
  if(getMemoryType()==BackEnd::Memory::Host)
    return m_internal->setValues(nrow, m_rows.data(), values);
  else
    return m_internal->setValuesOnDevice(nrow, m_rows.data(), values);
}


bool HypreVector::setValuesToZeros()
{
  if (m_internal == nullptr)
    return false;
  bool ok = m_internal->setValuesToZeros(m_rows.size(), m_rows.data()) ;
  if(ok)
    return m_internal->assemble();
  else
    return false ;
}

/*---------------------------------------------------------------------------*/

bool
HypreVector::getValues(const int nrow, const int* rows, double* values) const
{
  if (m_internal == nullptr)
    return false;
  return m_internal->getValues(nrow, rows, values);
}


bool
HypreVector::getValues(const int nrow, double* values) const
{
  if (m_internal == nullptr)
    return false;

  return m_internal->getValues(nrow, m_rows.data(), values);
}
/*---------------------------------------------------------------------------*/

bool
HypreVector::copyValuesToDevice(std::size_t nrows,
                                IndexType* rows_d,
                                ValueType* values_d) const
{
  if (m_internal == nullptr) return false ;
#ifdef ALIEN_USE_SYCL
  if(rows_d)
    return m_internal->getValuesToDevice(nrows, rows_d, values_d);
  else
    return m_internal->getValuesToDevice(nrows, m_rows.data(), values_d);
#else
  alien_fatal([&] {
    cout()<<"Error SYCL Support is required to get values of Hypre Vector from Device Memory";
  });
  return false ;
#endif
}

bool
HypreVector::copyValuesToHost(std::size_t nrows,
                              IndexType* rows_h,
                              ValueType* values_h) const
{
  if (m_internal == nullptr) return false ;
#ifdef ALIEN_USE_SYCL
  if(rows_h)
    return m_internal->getValuesToHost(nrows, rows_h, values_h);
  else
    return m_internal->getValuesToHost(nrows, m_rows.data(), values_h);
#else
  alien_fatal([&] {
    cout()<<"Error SYCL Support is required to get values of Hypre Vector from Device Memory";
  });
  return false ;
#endif
}

void HypreVector::allocateDevicePointers(std::size_t local_size, ValueType** values) const
{
  Internal::VectorInternal::allocateDevicePointers(local_size,values) ;
}

void HypreVector::freeDevicePointers(ValueType* values) const
{
  Internal::VectorInternal::freeDevicePointers(values) ;
}

/*---------------------------------------------------------------------------*/

bool
HypreVector::assemble()
{
  if (m_internal == nullptr)
    return false;
  return m_internal->assemble();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
HypreVector::update([[maybe_unused]] const HypreVector& v)
{
  ALIEN_ASSERT((this == &v), ("Unexpected error"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
