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

#include "to_simple_csr_vector.h"

#include <iostream>

#include <alien/kernels/redistributor/RedistributorBackEnd.h>
#include <alien/kernels/redistributor/RedistributorVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

namespace Alien
{

using namespace Arccore;

RedistributorToSimpleCSRVectorConverter::RedistributorToSimpleCSRVectorConverter() {}

RedistributorToSimpleCSRVectorConverter::~RedistributorToSimpleCSRVectorConverter() {}

BackEndId
RedistributorToSimpleCSRVectorConverter::sourceBackend() const
{
  return AlgebraTraits<BackEnd::tag::redistributor>::name();
}

BackEndId
RedistributorToSimpleCSRVectorConverter::targetBackend() const
{
  return AlgebraTraits<BackEnd::tag::simplecsr>::name();
}

void RedistributorToSimpleCSRVectorConverter::convert(
const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SrcVector& src = cast<SrcVector>(sourceImpl, sourceBackend());
  TgtVector& tgt = cast<TgtVector>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting : Redistributor " << &src << " to SimpleCSRVector " << &tgt;
  });

  if (sourceImpl->block()) {
    throw FatalErrorException(A_FUNCINFO, "Block matrices are not handled yet");
  }

  src.redistributeBack(tgt);
}

REGISTER_VECTOR_CONVERTER(RedistributorToSimpleCSRVectorConverter);

} // namespace Alien
