// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>

#include "alien/core/backend/IMatrixConverter.h"
#include "alien/core/backend/MatrixConverterRegisterer.h"
#include "alien/kernels/simple_csr/SimpleCSRMatrix.h"
#include "alien/kernels/simple_csr/SimpleCSRBackEnd.h"

#include "alien/kernels/mcg/data_structure/MCGMatrix.h"
#include "alien/kernels/mcg/MCGBackEnd.h"

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

class SimpleCSR_to_MCG_BCSR_MatrixConverter : public IMatrixConverter
{
 public:
  SimpleCSR_to_MCG_BCSR_MatrixConverter() = default;
  virtual ~SimpleCSR_to_MCG_BCSR_MatrixConverter() = default;

 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::mcgsolver>::name();
  }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  // void convert(const IMatrixImpl * sourceImpl, IMatrixImpl * targetImpl, int i, int j)
  // const;

  void _build(const SimpleCSRMatrix<Real>& sourceImpl,
    MCGMatrix<Real,MCGInternal::eMemoryDomain::Host>& targetImpl) const;
};

void
SimpleCSR_to_MCG_BCSR_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SimpleCSRMatrix<Real>& v =
    cast<SimpleCSRMatrix<Real>>(sourceImpl, sourceBackend());
  auto& v2 =
    cast<MCGMatrix<Real,MCGInternal::eMemoryDomain::Host>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRMatrix: " << &v << " to MCGMatrix " << &v2;
  });

  if (sourceImpl->vblock())
    throw FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

void
SimpleCSR_to_MCG_BCSR_MatrixConverter::_build(
    const SimpleCSRMatrix<Real>& sourceImpl,
    MCGMatrix<Real,MCGInternal::eMemoryDomain::Host>& targetImpl) const
{
  const MatrixDistribution& dist = targetImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const auto local_size = profile.getNRow();
  const auto global_size = dist.globalColSize();
  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();

  const SimpleCSRMatrix<Real>::MatrixInternal* matrixInternal = sourceImpl.internal();
  ConstArrayView<Real> values = matrixInternal->getValues();
  int block_size = 1;
  int block_size2 = 1;

  auto partition_offset = dist.rowOffset(dist.parallelMng()->commRank());

  if (sourceImpl.block()) {
    block_size = sourceImpl.block()->sizeX();
    block_size2 = sourceImpl.block()->sizeY();
  }

  if (!targetImpl.isInit()) {
    if (not targetImpl.initMatrix(MCGInternal::eMemoryDomain::Host,block_size, block_size2, local_size, global_size,
            row_offset.unguardedBasePointer(), cols.unguardedBasePointer(),
            partition_offset)) {
      throw FatalErrorException(A_FUNCINFO, "MCGSolver Initialisation failed");
    }
  }

  const bool success = targetImpl.initMatrixValues(MCGInternal::eMemoryDomain::Host,values.unguardedBasePointer());

  if (not success) {
    throw FatalErrorException(A_FUNCINFO, "Cannot set MCGSolver Matrix Values");
  }
}

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_MCG_BCSR_MatrixConverter);
