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

#include "to_simple_csr_matrix.h"

#include <alien/kernels/redistributor/RedistributorBackEnd.h>
#include <alien/kernels/redistributor/RedistributorMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

#include <alien/core/backend/MatrixConverterRegisterer.h>

namespace Alien
{

using namespace Arccore;

RedistributorToSimpleCSRMatrixConverter::RedistributorToSimpleCSRMatrixConverter() {}

RedistributorToSimpleCSRMatrixConverter::~RedistributorToSimpleCSRMatrixConverter() {}

BackEndId
RedistributorToSimpleCSRMatrixConverter::sourceBackend() const
{
  return AlgebraTraits<BackEnd::tag::redistributor>::name();
}

BackEndId
RedistributorToSimpleCSRMatrixConverter::targetBackend() const
{
  return AlgebraTraits<BackEnd::tag::simplecsr>::name();
}

void RedistributorToSimpleCSRMatrixConverter::convert(
const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SrcMatrix& src = cast<SrcMatrix>(sourceImpl, sourceBackend());
  TgtMatrix& tgt = cast<TgtMatrix>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting : Redistributor " << &src << " to SimpleCSRMatrix " << &tgt;
  });

  if (sourceImpl->block()) {
    throw FatalErrorException(A_FUNCINFO, "Block matrices are not handled yet");
  }
}

REGISTER_MATRIX_CONVERTER(RedistributorToSimpleCSRMatrixConverter);

} // namespace Alien
