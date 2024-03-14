// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#include <alien/core/impl/MultiMatrixImpl.h>

#include "alien/kernels/mcg/data_structure/MCGInternal.h"
#include "alien/kernels/mcg/data_structure/MCGMatrix.h"

BEGIN_MCGINTERNAL_NAMESPACE

END_MCGINTERNAL_NAMESPACE

namespace Alien {

MCGMatrix::MCGMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::mcgsolver>::name())
{
  m_internal = new MatrixInternal();
}

MCGMatrix::~MCGMatrix()
{
  delete m_internal;
}

bool
MCGMatrix::initMatrix(const int block_size, const int block_size2, const int nrow,
    int const* row_offset, int const* cols, int partition_offset)
{
  Integer nblocks = row_offset[nrow];
  std::shared_ptr<MCGInternal::MatrixInternal::ProfileType> profile(
      new MCGInternal::MatrixInternal::ProfileType(
          nrow, nrow, nblocks, partition_offset));

  m_internal->m_elem_perm.resize(nblocks);

  profile->rawSortInit(row_offset, cols, m_internal->m_elem_perm);

  profile->computeDiagIndex();

  m_internal->m_matrix.reset(
      new MCGInternal::MatrixInternal::MatrixType(block_size, block_size2, profile));

  m_internal->m_elliptic_split_tag = computeEllipticSplitTags(block_size);
  m_is_init = true;
  return true;
}

bool
MCGMatrix::initMatrixValues(Real const* values)
{
  m_internal->m_matrix->setValues(values, m_internal->m_elem_perm);
  return true;
}

bool
MCGMatrix::isInit() const
{
  return m_is_init;
}

bool
MCGMatrix::computeEllipticSplitTags(int equation_num) const
{
  bool elliptic_split_tag_found = false;
  const ISpace& space = this->rowSpace();
  const MatrixDistribution& dist = this->distribution();
  Integer min_local_index = dist.rowOffset();
  Integer local_size = dist.localRowSize();

  m_internal->m_equation_type = std::make_shared<MCGSolver::BVector<MCGSolver::Equation::eType>>
          (local_size * equation_num, 1);
  for (int i = 0; i < local_size * equation_num; ++i)
    m_internal->m_equation_type->data()[i] =
        MCGSolver::Equation::NoType; // NoTyp == 0 , Elliptic==1 cf. PrecondEquation.h

  for (Integer i = 0; i < space.nbField(); ++i) {
    const UniqueArray<Integer>& indices = space.field(i);
    if (space.fieldLabel(i) == "Elliptic" and not indices.empty()) {
      elliptic_split_tag_found = true;
      for (Integer j = 0; j < indices.size(); ++j) {
        const Integer index = indices[j];
        m_internal->m_equation_type->data()[(index - min_local_index) * equation_num] =
            MCGSolver::Equation::Elliptic; // NoTyp == 0 , Elliptic==1 cf.
                                           // PrecondEquation.h
      }
    }
  }

  return elliptic_split_tag_found;
}
} // namespace Alien
