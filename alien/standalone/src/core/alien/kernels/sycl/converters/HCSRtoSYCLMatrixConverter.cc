/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <iostream>

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/HCSRMatrix.h>
#include <alien/core/block/ComputeBlockOffsets.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>

#include "alien/kernels/sycl/data/SYCLBEllPackInternal.h"
#include "alien/kernels/sycl/data/HCSRMatrixInternal.h"

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class HCSRtoSYCLMatrixConverter : public IMatrixConverter
{
 public:
  HCSRtoSYCLMatrixConverter();
  virtual ~HCSRtoSYCLMatrixConverter() {}

 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hcsr>::name();
  }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::sycl>::name(); }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void _build(const HCSRMatrix<Real>& sourceImpl, SYCLBEllPackMatrix<Real>& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

HCSRtoSYCLMatrixConverter::HCSRtoSYCLMatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void HCSRtoSYCLMatrixConverter::convert(
const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const HCSRMatrix<Real>& v =
  cast<HCSRMatrix<Real>>(sourceImpl, sourceBackend());
  SYCLBEllPackMatrix<Real>& v2 = cast<SYCLBEllPackMatrix<Real>>(targetImpl, targetBackend());

  alien_debug(
  [&] { cout() << "Converting HCSRMatrix: " << &v << " to SYCLBEllPackMatrix " << &v2; });

  _build(v, v2);
}

void HCSRtoSYCLMatrixConverter::_build(
const HCSRMatrix<Real>& sourceImpl, SYCLBEllPackMatrix<Real>& targetImpl) const
{
  typedef HCSRMatrix<Real>::MatrixInternal HCSRMatrixType;

  const MatrixDistribution& dist = targetImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Integer globalSize = dist.globalRowSize();
  const Integer localOffset = dist.rowOffset();
  auto const& matrixInternal = *sourceImpl.internal();

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
                                  sourceImpl.getDistStructInfo())) {
      throw FatalErrorException(A_FUNCINFO, "SYCL Initialisation failed");
    }

    if (not targetImpl.internal()->setMatrixValues(matrixInternal.values())) {
      throw FatalErrorException(A_FUNCINFO, "Cannot set SYCL Matrix Values");
    }
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(HCSRtoSYCLMatrixConverter);
