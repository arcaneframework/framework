// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/core/block/ComputeBlockOffsets.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SimpleCSRtoSYCLMatrixConverter : public IMatrixConverter
{
 public:
  SimpleCSRtoSYCLMatrixConverter();
  virtual ~SimpleCSRtoSYCLMatrixConverter() {}

 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::sycl>::name(); }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void _build(const SimpleCSRMatrix<Real>& sourceImpl, SYCLBEllPackMatrix<Real>& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSRtoSYCLMatrixConverter::SimpleCSRtoSYCLMatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void SimpleCSRtoSYCLMatrixConverter::convert(
const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SimpleCSRMatrix<Real>& v =
  cast<SimpleCSRMatrix<Real>>(sourceImpl, sourceBackend());
  SYCLBEllPackMatrix<Real>& v2 = cast<SYCLBEllPackMatrix<Real>>(targetImpl, targetBackend());

  alien_debug(
  [&] { cout() << "Converting SimpleCSRMatrix: " << &v << " to SYCLBEllPackMatrix " << &v2; });

  _build(v, v2);
}

void SimpleCSRtoSYCLMatrixConverter::_build(
const SimpleCSRMatrix<Real>& sourceImpl, SYCLBEllPackMatrix<Real>& targetImpl) const
{
  typedef SimpleCSRMatrix<Real>::MatrixInternal CSRMatrixType;

  const MatrixDistribution& dist = targetImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Integer globalSize = dist.globalRowSize();
  const Integer localOffset = dist.rowOffset();
  auto const& matrixInternal = *sourceImpl.internal();

  auto block_size = sourceImpl.blockSize() ;

  {
    auto const& matrix_profile = sourceImpl.internal()->getCSRProfile();
    int nrows = matrix_profile.getNRow();
    int const* kcol = matrix_profile.getRowOffset().unguardedBasePointer();
    int const* cols = matrix_profile.getCols().unguardedBasePointer();

    if (not targetImpl.initMatrix(dist.parallelMng(),
                                  localOffset,
                                  globalSize,
                                  nrows,
                                  kcol,
                                  cols,
                                  sourceImpl.getDistStructInfo(),
                                  block_size)) {
      throw FatalErrorException(A_FUNCINFO, "SYCL Initialisation failed");
    }

    if (not targetImpl.setMatrixValues(matrixInternal.getDataPtr(), false)) {
      throw FatalErrorException(A_FUNCINFO, "Cannot set SYCL Matrix Values");
    }
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(SimpleCSRtoSYCLMatrixConverter);
