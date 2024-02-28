// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#include "from_simple_csr_matrix.h"

#include <alien/core/backend/MatrixConverterRegisterer.h>
#include <alien/kernels/dok/DoKBackEnd.h>
#include <alien/kernels/dok/DoKMatrixT.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

namespace Alien
{

using namespace Arccore;

SimpleCSRtoDoKMatrixConverter::SimpleCSRtoDoKMatrixConverter() {}

SimpleCSRtoDoKMatrixConverter::~SimpleCSRtoDoKMatrixConverter() {}

BackEndId
SimpleCSRtoDoKMatrixConverter::sourceBackend() const
{
  return AlgebraTraits<BackEnd::tag::simplecsr>::name();
}

BackEndId
SimpleCSRtoDoKMatrixConverter::targetBackend() const
{
  return AlgebraTraits<BackEnd::tag::DoK>::name();
}

void SimpleCSRtoDoKMatrixConverter::convert(
const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SrcMatrix& src = cast<SrcMatrix>(sourceImpl, sourceBackend());
  TgtMatrix& tgt = cast<TgtMatrix>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRMatrix: " << &src << " to DoKMatrix " << &tgt;
  });

  if (sourceImpl->block()) {
    throw FatalErrorException(A_FUNCINFO, "Block matrices are not handled yet");
  }

  this->_build(src, tgt);
}

void SimpleCSRtoDoKMatrixConverter::_build(const SrcMatrix& src, TgtMatrix& tgt) const
{
  const MatrixDistribution& dist = tgt.distribution();

  const SimpleCSRInternal::CSRStructInfo& profile = src.getCSRProfile();
  const Integer localSize = profile.getNRow();
  const Integer localOffset = dist.rowOffset();

  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = *src.internal();

  alien_debug([&] {
    cout() << "Matrix range : [" << localOffset << ":" << localOffset + localSize - 1
           << "]";
  });

  ConstArrayView<Integer> cols = profile.getCols();
  ConstArrayView<Real> csr_values = matrixInternal.getValues();
  for (Integer irow = 0; irow < localSize; ++irow) {
    int row = localOffset + irow;
    int ncols = profile.getRowSize(irow);
    Integer icount = profile.getRowOffset()[irow];
    for (Integer k = 0; k < ncols; ++k) {
      tgt.setNNZ(row, cols[icount + k], csr_values[icount + k]);
    }
  }
}

REGISTER_MATRIX_CONVERTER(SimpleCSRtoDoKMatrixConverter);

} // namespace Alien
