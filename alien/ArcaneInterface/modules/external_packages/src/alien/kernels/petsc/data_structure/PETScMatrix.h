// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once


#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/kernels/petsc/PETScPrecomp.h>
#include <alien/core/impl/IMatrixImpl.h>
#include <alien/core/backend/BackEnd.h>

#include <alien/data/ISpace.h>
#include <alien/ref/data/scalar/Vector.h>
#include <alien/ref/handlers/scalar/VectorReader.h>

/*---------------------------------------------------------------------------*/

namespace Alien::PETScInternal {

struct MatrixInternal;
}

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

class ALIEN_EXTERNAL_PACKAGES_EXPORT PETScMatrix : public IMatrixImpl
{
 public:
  typedef PETScInternal::MatrixInternal MatrixInternal;

 public:
  PETScMatrix(const MultiMatrixImpl* multi_impl);
  virtual ~PETScMatrix();

  BackEnd::Memory::eType getMemoryType() const ;

  BackEnd::Exec::eSpaceType getExecSpace() const ;

 public:
  void clear() {}

 public:
  bool initMatrix(const int local_size,
                  const int local_offset,
                  const int global_size,
                  const int block_size,
                  Arccore::ConstArrayView<Arccore::Integer> diag_lineSizes,
                  Arccore::ConstArrayView<Arccore::Integer> offdiag_lineSizes,
                  const bool parallel);

  bool initMatrix(const int local_size,
                  const int local_offset,
                  const int global_size,
                  const int block_size,
                  const int nb_dofs,
                  int const* dof_uids,
                  const int nnz,
                  int* rows,
                  int* cols,
                  const bool parallel) ;

  bool addMatrixValues(
      const int row, const int ncols, const int* cols, const Arccore::Real* values);

  bool setMatrixValues(
      const int row, const int ncols, const int* cols, const Arccore::Real* values);

  bool setMatrixValuesFromCSR(const Arccore::Real* values);

#ifdef PETSC_HAVE_MATSETBLOCKSIZE
  bool addMatrixBlockValues(
      const int row, const int ncols, const int* cols, const Arccore::Real* values);

  bool setMatrixBlockValues(
      const int row, const int ncols, const int* cols, const Arccore::Real* values);
#endif

  bool setInitValues(const int nrow, const int* rows, const Arccore::Real* values);

  void setMatrixCoordinate(Vector const& x, Vector const& y, Vector const& z) ;
  void setMatrixCoordinate(Vector const& x, Vector const& y) ;

  bool assemble();

 public:
  MatrixInternal* internal() ;
  const MatrixInternal* internal() const ;

 private:
  Arccore::Integer ijk(Arccore::Integer i, Arccore::Integer j, Arccore::Integer k,
      Arccore::Integer block_size, Arccore::Integer unknowns_num) const
  {
    return k * block_size + i * unknowns_num + j;
  }

  std::unique_ptr<MatrixInternal> m_internal ;

  Arccore::MessagePassing::IMessagePassingMng* m_pm = nullptr;
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/

