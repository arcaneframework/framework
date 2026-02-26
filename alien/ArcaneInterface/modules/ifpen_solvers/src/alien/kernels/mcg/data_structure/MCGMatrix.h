// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "alien/core/impl/IMatrixImpl.h"
#include "alien/core/block/Block.h"
#include "alien/data/Space.h"

#include "alien/kernels/mcg/MCGPrecomp.h"
#include "alien/kernels/mcg/MCGBackEnd.h"
#include "alien/kernels/mcg/data_structure/MemoryDomain.h"

BEGIN_MCGINTERNAL_NAMESPACE

template<typename NumT,eMemoryDomain Domain>
class MatrixInternal;
END_MCGINTERNAL_NAMESPACE

namespace Alien {

template<typename NumT,MCGInternal::eMemoryDomain Domain>
class MCGMatrix : public IMatrixImpl
{
 public:
  using MatrixInternal = MCGInternal::MatrixInternal<NumT,Domain>;

  MCGMatrix(const MultiMatrixImpl* multi_impl);
  ~MCGMatrix() override = default;

  void init(
      const ISpace& row_space, const ISpace& col_space, const MatrixDistribution& dist)
  {
    std::cout << "init MCGMatrix with m_domain_offset = dist.rowOffset()/m_equations_num "
              << std::endl;
  }

  void initSpace0(const Space& space) { m_space0 = &space; }

  void initSpace1(const Space& space) { m_space1 = &space; }

  const ISpace& space() const
  {
    if (m_space0)
      return *m_space0;
    else
      return IMatrixImpl::rowSpace();
  }

  const Space& space0() const { return *m_space0; }

  const Space& space1() const { return *m_space1; }

  void clear() {}

  bool isInit() const;

  //! Ensemble des tags pour la construction CprAMG
  bool computeEllipticSplitTags(int equation_num) const;

 public:
  bool initMatrix(const MCGInternal::eMemoryDomain src_domain,const int block_size, const int block_size2,
      const int nrow,const int ncol,
      int const* row_offset, int const* cols, int partition_offset);

  bool initMatrixValues(const MCGInternal::eMemoryDomain src_domain,Real const* values);

  Space m_row_space1;
  Space m_col_space1;

  MatrixInternal* internal() { return m_internal.get(); }
  const MatrixInternal* internal() const { return m_internal.get(); }

 private:
  std::unique_ptr<MatrixInternal> m_internal = std::make_unique<MatrixInternal>();
  Space const* m_space0 = nullptr;
  Space const* m_space1 = nullptr;
  bool m_is_init = false;
};

}

