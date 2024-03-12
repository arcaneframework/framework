// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/hts/data_structure/HTSMatrix.h>
#include <alien/core/block/ComputeBlockOffsets.h>

#include <alien/kernels/hts/HTSBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_HTS_MatrixConverter : public IMatrixConverter
{
 public:
  SimpleCSR_to_HTS_MatrixConverter();
  virtual ~SimpleCSR_to_HTS_MatrixConverter() {}
 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::hts>::name(); }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void _build(const SimpleCSRMatrix<Real>& sourceImpl, HTSMatrix<Real>& targetImpl) const;
  void _buildBlock(
      const SimpleCSRMatrix<Real>& sourceImpl, HTSMatrix<Real>& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_HTS_MatrixConverter::SimpleCSR_to_HTS_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_HTS_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SimpleCSRMatrix<Real>& v =
      cast<SimpleCSRMatrix<Real>>(sourceImpl, sourceBackend());
  HTSMatrix<Real>& v2 = cast<HTSMatrix<Real>>(targetImpl, targetBackend());

  alien_debug(
      [&] { cout() << "Converting SimpleCSRMatrix: " << &v << " to HTSMatrix " << &v2; });

  if (targetImpl->block())
    _buildBlock(v, v2);
  else if (targetImpl->vblock())
    throw FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

void
SimpleCSR_to_HTS_MatrixConverter::_build(
    const SimpleCSRMatrix<Real>& sourceImpl, HTSMatrix<Real>& targetImpl) const
{
  typedef SimpleCSRMatrix<Real>::MatrixInternal CSRMatrixType;

  const MatrixDistribution& dist = targetImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Integer localSize = profile.getNRow();
  const Integer localOffset = dist.rowOffset();
  auto const& matrixInternal = *sourceImpl.internal();
  const Integer myRank = dist.parallelMng()->commRank();
  const Integer nProc = dist.parallelMng()->commSize();

  {

    auto const& matrix_profile = sourceImpl.internal()->getCSRProfile();
    int nrows = matrix_profile.getNRow();
    int const* kcol = matrix_profile.getRowOffset().unguardedBasePointer();
    int const* cols = matrix_profile.getCols().unguardedBasePointer();
    int block_size = sourceImpl.block() ? sourceImpl.block()->size() : 1;

    if (not targetImpl.initMatrix(dist.parallelMng(), nrows, kcol, cols, block_size)) {
      throw FatalErrorException(A_FUNCINFO, "HTS Initialisation failed");
    }

    if (not targetImpl.setMatrixValues(matrixInternal.getDataPtr())) {
      throw FatalErrorException(A_FUNCINFO, "Cannot set HTS Matrix Values");
    }

    if (not targetImpl.computeDDMatrix()) {
      throw FatalErrorException(A_FUNCINFO, "Cannot set HTS DDMatrix Values");
    }
  }
}

void
SimpleCSR_to_HTS_MatrixConverter::_buildBlock(
    const SimpleCSRMatrix<Real>& sourceImpl, HTSMatrix<Real>& targetImpl) const
{
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_HTS_MatrixConverter);
