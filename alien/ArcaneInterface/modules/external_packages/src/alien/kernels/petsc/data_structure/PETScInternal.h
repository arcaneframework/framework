// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/kernels/petsc/PETScPrecomp.h>

#include "petscmat.h"
#include "petscvec.h"

// For old PETSc version
#ifndef MATSOLVERSUPERLU_DIST
#define MATSOLVERSUPERLU_DIST MAT_SOLVER_SUPERLU_DIST
#endif // MATSOLVERSUPERLU_DIST

#ifndef MATSOLVERSUPERLU
#define MATSOLVERSUPERLU MAT_SOLVER_SUPERLU
#endif // MATSOLVERSUPERLU

//! Internal struct for PETSc implementation
/*! Separate data from header;
 *  can be only included by PETSc implementations
 */

#include <alien/core/backend/BackEnd.h>
/*---------------------------------------------------------------------------*/

namespace Alien::PETScInternal {

/*---------------------------------------------------------------------------*/

struct MatrixInternal
{
  typedef Mat matrix_type;

 public:
  MatrixInternal(int local_offset,
                 int local_size,
                 int block_size,
                 bool parallel,
                 BackEnd::Memory::eType memory_type,
                 BackEnd::Exec::eSpaceType exec_space)
  : m_internal()
  , m_type(mat_type(parallel,memory_type,exec_space))
  , m_offset(local_offset)
  , m_local_size(local_size)
  , m_block_size(block_size)
  , m_parallel(parallel)
  , m_memory_type(memory_type)
  , m_exec_space(exec_space)
  {
  }

  ~MatrixInternal()
  {
    if (m_internal) {
//#ifndef PETSC_MATDESTROY_NEW
//      MatDestroy(m_internal);
//#else /* PETSC_MATDESTROY_NEW */
      MatDestroy(&m_internal);
//#endif /* PETSC_MATDESTROY_NEW */
    }
  }

  const MatType mat_type(bool parallel,
                     BackEnd::Memory::eType memory_type,
                     BackEnd::Exec::eSpaceType exec_space)
  {
    switch(exec_space)
    {
      case BackEnd::Exec::Device:
      {
#if PETSC_VERSION_GE(3, 20, 0)
#if PETSC_HAVE_CUDA
        return MatType(parallel ? MATMPIAIJCUSPARSE : MATSEQAIJCUSPARSE) ;
#else
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSC Matrix Type for CUDA Execution is not available");
#endif
#else
        throw Arccore::FatalErrorException(A_FUNCINFO, "PETSC Matrix Type for Device Execution is not available");
#endif
      }
      break ;
      case BackEnd::Exec::Host:
      default:
      {
        return MatType(parallel ? MATMPIAIJ : MATSEQAIJ) ;
      }
    }
  }

  BackEnd::Memory::eType getMemoryType() const {
    return m_memory_type ;
  }

  BackEnd::Exec::eSpaceType getExecSpace() const {
    return m_exec_space  ;
  }

 public:
  Mat m_internal;
  const MatType m_type;
  int m_offset = 0 ;
  Integer m_local_size = 0 ;
  Integer m_block_size = 1 ;
  bool m_parallel = false;
  Alien::BackEnd::Memory::eType m_memory_type = BackEnd::Memory::Host ;
  Alien::BackEnd::Exec::eSpaceType m_exec_space = BackEnd::Exec::Host ;

  bool m_has_coordinates = false ;
  Integer m_coordinates_dim = 3;
  Vec m_coordinates;

};

/*---------------------------------------------------------------------------*/

struct VectorInternal
{
 public:
  typedef Vec vector_type;
  typedef Integer IndexType ;
  typedef Integer UIDIndexType ;
  typedef Integer LIDIndexType ;

 public:
  VectorInternal(const int local_size,
                 const int local_offset,
                 const int global_size,
                 const bool parallel,
                 MPI_Comm comm,
                 BackEnd::Memory::eType memory_type,
                 BackEnd::Exec::eSpaceType exec_space);

  VectorInternal(const int local_size,
                 const int local_offset,
                 const int global_size,
                 const int block_size,
                 const bool parallel,
                 MPI_Comm comm,
                 BackEnd::Memory::eType memory_type,
                 BackEnd::Exec::eSpaceType exec_space);
  ~VectorInternal();

  BackEnd::Memory::eType getMemoryType() const {
    return m_memory_type ;
  }

  BackEnd::Exec::eSpaceType getExecSpace() const {
    return m_exec_space  ;
  }

  bool memoryOnHost() const {
    return m_memory_type == BackEnd::Memory::Host ;
  }

  bool memoryOnDevice() const {
    return m_memory_type == BackEnd::Memory::Device ;
  }

 public:
  Vec m_internal;
  int m_offset = 0 ;
  Integer m_local_size = 0 ;
  bool m_parallel = false;
  Alien::BackEnd::Memory::eType m_memory_type = BackEnd::Memory::Host ;
  Alien::BackEnd::Exec::eSpaceType m_exec_space = BackEnd::Exec::Host ;
};

/*---------------------------------------------------------------------------*/

} // namespace Alien::PETScInternal

/*---------------------------------------------------------------------------*/
