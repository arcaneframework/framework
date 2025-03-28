﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <vector>
#include "PETScVector.h"
/* Author : havep at Wed Jul 18 14:08:21 2012
 * Generated by createNew
 */

#include <alien/kernels/petsc/PETScBackEnd.h>
#include <alien/kernels/petsc/data_structure/PETScInternal.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

PETScVector::PETScVector(const MultiVectorImpl* multi_impl)
: IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::petsc>::name())
{
  ;
}

/*---------------------------------------------------------------------------*/

PETScVector::~PETScVector()
{}

/*---------------------------------------------------------------------------*/

void
PETScVector::init([[maybe_unused]] const VectorDistribution& dist,
                  const bool need_allocate, [[maybe_unused]] Arccore::Integer block_size)
{
  if (need_allocate)
    allocate();
}

/*---------------------------------------------------------------------------*/

void
PETScVector::allocate()
{
  const VectorDistribution& dist = this->distribution();

  Arccore::MessagePassing::Mpi::MpiMessagePassingMng*
  mpi_pm = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(dist.parallelMng()) ;
  MPI_Comm comm ;

  if(mpi_pm && mpi_pm->getMPIComm())
    comm = *mpi_pm->getMPIComm() ;
  else
    comm = MPI_COMM_NULL ;

  auto blk = this->block() ;
  if(blk)
  {
      const Arccore::Integer block_size = blk->size();
      m_internal.reset(new VectorInternal(this->scalarizedLocalSize(),
                                          this->scalarizedOffset(),
                                          this->scalarizedGlobalSize(),
                                          block_size,
                                          dist.isParallel(),
                                          comm));
  }
  else
  {
      m_internal.reset(new VectorInternal(this->scalarizedLocalSize(), this->scalarizedOffset(),
                                          this->scalarizedGlobalSize(), dist.isParallel(),comm));
  }
}

/*---------------------------------------------------------------------------*/

bool
PETScVector::setValues(const int nrow, const int* rows, const double* values)
{
  if (m_internal->m_internal == nullptr)
    return false;
  int ierr = VecSetValues(m_internal->m_internal,
      nrow, // nb de valeurs
      rows, values, INSERT_VALUES);
  return (ierr == 0);
}

bool
PETScVector::setBlockValues(const int nrow, const int* rows,[[maybe_unused]] const int block_size, const double* values)
{
  if (m_internal->m_internal == nullptr)
    return false;
  int ierr = VecSetValuesBlocked(m_internal->m_internal,
      nrow, // nb de valeurs
      rows, values, INSERT_VALUES);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
PETScVector::setValues(const int nrow, const double* values)
{
  if (!m_internal.get())
    return false;

  std::vector<int> rows(nrow);
  for (int i = 0; i < nrow; ++i)
    rows[i] = m_internal->m_offset + i;
  int ierr = VecSetValues(m_internal->m_internal,
      nrow, // nb de valeurs
      rows.data(), values, INSERT_VALUES);
  return (ierr == 0);
}

bool
PETScVector::setBlockValues(const int nrow,[[maybe_unused]] const int block_size, const double* values)
{
  if (!m_internal.get())
    return false;

  std::vector<int> rows(nrow);
  for (int i = 0; i < nrow; ++i)
    rows[i] = m_internal->m_offset + i;
  int ierr = VecSetValuesBlocked(m_internal->m_internal,
      nrow, // nb de valeurs
      rows.data(), values, INSERT_VALUES);
  return (ierr == 0);
}
/*---------------------------------------------------------------------------*/

bool
PETScVector::setValues(Arccore::ConstArrayView<Arccore::Real> values)
{
  ALIEN_ASSERT((m_internal.get()), ("Not initialized PETScVector before updating"));
  if (not setValues(this->scalarizedLocalSize(), values.unguardedBasePointer()))
    throw Arccore::FatalErrorException(A_FUNCINFO, "Error while setting vetor data");
  if (not assemble())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Error while assembling vector data");
  return true;
}


bool
PETScVector::setBlockValues([[maybe_unused]] int block_size, Arccore::ConstArrayView<Arccore::Real> values)
{
  ALIEN_ASSERT((m_internal.get()), ("Not initialized PETScVector before updating"));
#ifdef  PETSC_HAVE_VECSETBLOCKSIZE
  if (not setBlockValues(this->scalarizedLocalSize()/block_size, block_size, values.unguardedBasePointer()))
    throw Arccore::FatalErrorException(A_FUNCINFO, "Error while setting vetor data");
#else
  if (not setValues(this->scalarizedLocalSize(), values.unguardedBasePointer()))
    throw Arccore::FatalErrorException(A_FUNCINFO, "Error while setting vetor data");
#endif
  if (not assemble())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Error while assembling vector data");
  return true;
}

/*---------------------------------------------------------------------------*/

bool
PETScVector::getValues(const int nrow, const int* rows, double* values) const
{
  if (!m_internal.get())
    return false;

  int ierr = VecGetValues(m_internal->m_internal,
      nrow, // nb de valeurs
      rows, values);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
PETScVector::getValues(const int nrow, double* values) const
{
  if (!m_internal.get())
    return false;

  int* rows = new int[nrow];
  for (int i = 0; i < nrow; ++i)
    rows[i] = m_internal->m_offset + i;
  int ierr = VecGetValues(m_internal->m_internal,
      nrow, // nb de valeurs
      rows, values);
  delete[] rows;
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
PETScVector::assemble()
{
  int ierr = 0;
  ierr += VecAssemblyBegin(m_internal->m_internal);
  ierr += VecAssemblyEnd(m_internal->m_internal);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
void
PETScVector::
update(const SimpleCSRVector<double> & v)
{
  ALIEN_ASSERT((m_internal!=NULL),("Not initialized PETScVector before updating"));
  ConstArrayView<Real> values = v.values();
  if (not setValues(m_space->localSize()*m_block_size,values.unguardedBasePointer()))
    throw FatalErrorException(A_FUNCINFO, "Error while setting vetor data");
  if (not assemble())
    throw FatalErrorException(A_FUNCINFO, "Error while assembling vector data");
}
*/
/*---------------------------------------------------------------------------*/
/*
void
PETScVector::
update(const PETScVector & v)
{
  ALIEN_ASSERT((this == &v),("Unexpected error"));
}
*/
/*---------------------------------------------------------------------------*/
/*
void
PETScVector::
update(const IFPVector & v)
{
  UniqueArray<Real> values(m_space->localSize()*m_block_size);
  v.getValues(values.size(),values.unguardedBasePointer()) ;
  if (not setValues(values.size(),values.unguardedBasePointer()))
    throw FatalErrorException(A_FUNCINFO, "Error while setting vetor data");
  if (not assemble())
    throw FatalErrorException(A_FUNCINFO, "Error while assembling vector data");
}
*/
/*---------------------------------------------------------------------------*/
/*
void
PETScVector::
update(const MTLVector & v)
{
  UniqueArray<Real> values(m_space->localSize()*m_block_size);
  v.getValues(values.size(),values.unguardedBasePointer()) ;
  if (not setValues(values.size(),values.unguardedBasePointer()))
    throw FatalErrorException(A_FUNCINFO, "Error while setting vetor data");
  if (not assemble())
    throw FatalErrorException(A_FUNCINFO, "Error while assembling vector data");
}

void
PETScVector::
update(const HypreVector & v)
{
  UniqueArray<Real> values(m_space->localSize()*m_block_size);
  v.getValues(values.size(),values.unguardedBasePointer()) ;
  if (not setValues(values.size(),values.unguardedBasePointer()))
    throw FatalErrorException(A_FUNCINFO, "Error while setting vetor data");
  if (not assemble())
    throw FatalErrorException(A_FUNCINFO, "Error while assembling vector data");
}


void
PETScVector::
update(const MCGVector & v)
{
  UniqueArray<Real> values(m_space->localSize()*m_block_size);
  v.getValues(values.size(),values.unguardedBasePointer()) ;
  if (not setValues(values.size(),values.unguardedBasePointer()))
    throw FatalErrorException(A_FUNCINFO, "Error while setting vetor data");
  if (not assemble())
    throw FatalErrorException(A_FUNCINFO, "Error while assembling vector data");
}
*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
