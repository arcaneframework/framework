// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "PETScInternal.h"

/*---------------------------------------------------------------------------*/

namespace Alien::PETScInternal {

/*---------------------------------------------------------------------------*/

VectorInternal::VectorInternal(const int local_size, const int local_offset,
    const int global_size, const bool parallel, MPI_Comm comm)
: m_offset(local_offset)
, m_parallel(parallel)
{
  int ierr = 0;
  if (m_parallel) { // Use parallel structures
    // -- B Vector --
    ierr += VecCreateMPI(comm, local_size, global_size, &m_internal);
  } else { // Use sequential structures
    // -- B Vector --
    ierr += VecCreateSeq(PETSC_COMM_SELF, local_size, &m_internal);
  }
  int low;
  ierr += VecGetOwnershipRange(m_internal, &low, nullptr);

  if (low != local_offset)
    throw Arccore::FatalErrorException(A_FUNCINFO, "Ill placed parallel vector");
}


VectorInternal::VectorInternal(const int local_size,
                               const int local_offset,
                               const int global_size,
                               const int block_size,
                               const bool parallel,
                               MPI_Comm comm)
: m_offset(local_offset)
, m_parallel(parallel)
{
  int ierr = 0;
  if (m_parallel) { // Use parallel structures
    // -- B Vector --
    ierr += VecCreateMPI(comm, local_size, global_size, &m_internal);
  } else { // Use sequential structures
    // -- B Vector --
    ierr += VecCreateSeq(PETSC_COMM_SELF, local_size, &m_internal);
  }
#ifdef PETSC_HAVE_VECSETBLOCKSIZE
  ierr += VecSetBlockSize(m_internal,block_size) ;
#endif
  int low;
  ierr += VecGetOwnershipRange(m_internal, &low, nullptr);

  if (low != local_offset)
    throw Arccore::FatalErrorException(A_FUNCINFO, "Ill placed parallel vector");
  if(ierr)
    throw Arccore::FatalErrorException(A_FUNCINFO, "Errors while creating vector");
}


/*---------------------------------------------------------------------------*/

VectorInternal::~VectorInternal()
{
#ifndef PETSC_DESTROY_NEW
  VecDestroy(m_internal);
#else /* PETSC_DESTROY_NEW */
  VecDestroy(&m_internal);
#endif /* PETSC_DESTROY_NEW */
}

/*---------------------------------------------------------------------------*/

} // namespace Alien::PETScInternal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
