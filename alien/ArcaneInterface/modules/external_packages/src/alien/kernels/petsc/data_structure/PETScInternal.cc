// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#include <alien/core/backend/BackEnd.h>

#include "PETScInternal.h"

/*---------------------------------------------------------------------------*/

namespace Alien::PETScInternal {

/*---------------------------------------------------------------------------*/

VectorInternal::VectorInternal(const int local_size,
                               const int local_offset,
                               const int global_size,
                               const bool parallel,
                               MPI_Comm comm,
                               BackEnd::Memory::eType memory_type,
                               BackEnd::Exec::eSpaceType exec_space)
: m_offset(local_offset)
, m_parallel(parallel)
, m_memory_type(memory_type)
, m_exec_space(exec_space)

{
  int ierr = 0;
  switch(m_exec_space)
  {
    case BackEnd::Exec::Device:
    {
#if PETSC_VERSION_GE(3, 20, 0)
#if PETSC_USE_CUDA
      if (m_parallel) { // Use parallel structures
        ierr += VecCreateMPICUDA(comm, local_size, global_size, &m_internal);
      } else {
        ierr += VecCreateSeqCUDA(PETSC_COMM_SELF, local_size, &m_internal);
      }
#else
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSC Vector Type for CUDA Execution is not available");
#endif
#else
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSC Vector Type for Device Execution is not available");
#endif
    }
    break;
    case BackEnd::Exec::Host:
    default:
    {
      if (m_parallel) { // Use parallel structures
        // -- B Vector --
        ierr += VecCreateMPI(comm, local_size, global_size, &m_internal);
      } else { // Use sequential structures
        // -- B Vector --
        ierr += VecCreateSeq(PETSC_COMM_SELF, local_size, &m_internal);
      }
    }
  }

  int low;
  ierr += VecGetOwnershipRange(m_internal, &low, nullptr);

  if (low != local_offset)
    throw Arccore::FatalErrorException(A_FUNCINFO, "Ill placed parallel vector");
}


VectorInternal::VectorInternal(const int local_size,
                               const int local_offset,
                               const int global_size,
                               [[maybe_unused]] const int block_size,
                               const bool parallel,
                               MPI_Comm comm,
                               BackEnd::Memory::eType memory_type,
                               BackEnd::Exec::eSpaceType exec_space)
: m_offset(local_offset)
, m_parallel(parallel)
, m_memory_type(memory_type)
, m_exec_space(exec_space)
{
  int ierr = 0;
  switch(m_exec_space)
  {
    case BackEnd::Exec::Device:
    {
#if PETSC_VERSION_GE(3, 20, 0)
#if PETSC_USE_CUDA
      if (m_parallel) { // Use parallel structures
        ierr += VecCreateMPICUDA(comm, local_size, global_size, &m_internal);
      } else {
        ierr += VecCreateSeqCUDA(PETSC_COMM_SELF, local_size, &m_internal);
      }
#else
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSC CUDA Vector Type is not available");
#endif
#else
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSC Vector Type for Device Execution is not available");
#endif
    }
    break ;
    case BackEnd::Exec::Host:
    default:
    {
        if (m_parallel) { // Use parallel structures
          // -- B Vector --
          ierr += VecCreateMPI(comm, local_size, global_size, &m_internal);
        } else { // Use sequential structures
          // -- B Vector --
          ierr += VecCreateSeq(PETSC_COMM_SELF, local_size, &m_internal);
        }
    }
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
//#ifndef PETSC_DESTROY_NEW
//  VecDestroy(m_internal);
//#else /* PETSC_DESTROY_NEW */
  VecDestroy(&m_internal);
//#endif /* PETSC_DESTROY_NEW */
}

/*---------------------------------------------------------------------------*/

} // namespace Alien::PETScInternal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
