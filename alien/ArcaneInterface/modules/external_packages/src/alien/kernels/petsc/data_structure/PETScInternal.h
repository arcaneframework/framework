// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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

/*---------------------------------------------------------------------------*/

namespace Alien::PETScInternal {

/*---------------------------------------------------------------------------*/

struct MatrixInternal
{
  typedef Mat matrix_type;

 public:
  MatrixInternal(int local_offset, int local_size, int block_size, bool parallel)
  : m_internal()
  , m_type((parallel) ? MATMPIAIJ : MATSEQAIJ)
  , m_offset(local_offset)
  , m_local_size(local_size)
  , m_block_size(block_size)
  , m_parallel(parallel)
  {
  }

  ~MatrixInternal()
  {
    if (m_internal) {
#ifndef PETSC_MATDESTROY_NEW
      MatDestroy(m_internal);
#else /* PETSC_MATDESTROY_NEW */
      MatDestroy(&m_internal);
#endif /* PETSC_MATDESTROY_NEW */
    }
  }


 public:
  Mat m_internal;
  const MatType m_type;
  int m_offset = 0 ;
  Integer m_local_size = 0 ;
  Integer m_block_size = 1 ;
  bool m_parallel = false;
  bool m_has_coordinates = false ;
  Integer m_coordinates_dim = 3;
  Vec m_coordinates;

};

/*---------------------------------------------------------------------------*/

struct VectorInternal
{
 public:
  typedef Vec vector_type;

 public:
  VectorInternal(const int local_size,
                 const int local_offset,
                 const int global_size,
                 const bool parallel,
                 MPI_Comm comm);

  VectorInternal(const int local_size,
                 const int local_offset,
                 const int global_size,
                 const int block_size,
                 const bool parallel,
                 MPI_Comm comm);
  ~VectorInternal();

 public:
  Vec m_internal;
  int m_offset = 0 ;
  Integer m_local_size = 0 ;
  bool m_parallel = false;
};

/*---------------------------------------------------------------------------*/

} // namespace Alien::PETScInternal

/*---------------------------------------------------------------------------*/
