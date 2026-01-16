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
}

/*---------------------------------------------------------------------------*/

bool
HypreVector::setValues(const int nrow, const int* rows, const double* values)
{
  if (m_internal == NULL)
    return false;
  return m_internal->setValues(nrow, rows, values);
}

/*---------------------------------------------------------------------------*/

bool
HypreVector::setValues([[maybe_unused]] const int nrow, const double* values)
{
  if (m_internal == NULL)
    return false;

  return m_internal->setValues(m_rows.size(), m_rows.data(), values);
}

/*---------------------------------------------------------------------------*/

bool
HypreVector::getValues(const int nrow, const int* rows, double* values) const
{
  if (m_internal == NULL)
    return false;
  return m_internal->getValues(nrow, rows, values);
}

/*---------------------------------------------------------------------------*/

bool
HypreVector::getValues([[maybe_unused]] const int nrow, double* values) const
{
  if (m_internal == NULL)
    return false;
  if(m_internal->getMemoryType()==Alien::BackEnd::Memory::Host)
    return m_internal->getValues(m_rows.size(), m_rows.data(), values);
  else
  {
#ifdef ALIEN_USE_SYCL
      Alien::HypreVector::IndexType* rows_d = nullptr;
      Alien::HypreVector::ValueType* values_d = nullptr ;
      Alien::SYCLVector<Arccore::Real>::allocateDevicePointers(m_rows.size(),
                                                               &rows_d,
                                                               &values_d) ;
      m_internal->getValues(m_rows.size(), rows_d, values_d);
      Alien::SYCLVector<Arccore::Real>::copyDeviceToHost(m_rows.size(), values_d, values) ;
      Alien::SYCLVector<Arccore::Real>::freeDevicePointers(rows_d, values_d) ;
      return true ;
#else
      alien_fatal([&] {
        cout()<<"Error SYCL Support is required to get values of Hypre Vector from Device Memory";
      });
      return false ;
#endif
  }
}

/*---------------------------------------------------------------------------*/

bool
HypreVector::assemble()
{
  if (m_internal == NULL)
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
