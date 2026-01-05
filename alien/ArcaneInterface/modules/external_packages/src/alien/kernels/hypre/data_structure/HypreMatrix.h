// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ALIEN_KERNELS_HYPRE_DATASTRUCTURE_HYPREMATRIX_H
#define ALIEN_KERNELS_HYPRE_DATASTRUCTURE_HYPREMATRIX_H

#include <alien/AlienExternalPackagesPrecomp.h>
#include <alien/core/impl/IMatrixImpl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Internal {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  class MatrixInternal;

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ALIEN_EXTERNAL_PACKAGES_EXPORT HypreMatrix : public IMatrixImpl
{
 public:
  typedef Internal::MatrixInternal MatrixInternal;

 public:
  HypreMatrix(const MultiMatrixImpl* multi_impl);
  virtual ~HypreMatrix();

  BackEnd::Memory::eType getMemoryType() const ;

  BackEnd::Exec::eSpaceType getExecSpace() const ;

 public:
  void clear() {}

 public:
  bool initMatrix(const int ilower, const int iupper, const int jlower, const int jupper,
      const Arccore::ConstArrayView<Arccore::Integer>& lineSizes);

  bool addMatrixValues(const int nrow, const int* rows, const int* ncols, const int* cols,
      const Arccore::Real* values);

  bool setMatrixValues(const int nrow, const int* rows, const int* ncols, const int* cols,
      const Arccore::Real* values);

  bool assemble();

 public:
  MatrixInternal* internal() { return m_internal; }
  const MatrixInternal* internal() const { return m_internal; }

  Arccore::MessagePassing::IMessagePassingMng* getParallelMng() const { return m_pm; }

 private:
  Arccore::Integer ijk(Arccore::Integer i, Arccore::Integer j, Arccore::Integer k,
      Arccore::Integer block_size, Arccore::Integer unknowns_num) const
  {
    return k * block_size + i * unknowns_num + j;
  }

  MatrixInternal* m_internal = nullptr;
  Arccore::MessagePassing::IMessagePassingMng* m_pm = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_KERNELS_HYPRE_DATASTRUCTURE_HYPREMATRIX_H */
