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

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <alien/kernels/sycl/data/SYCLVector.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class SYCLtoSimpleCSRVectorConverter : public IVectorConverter
{
 public:
  SYCLtoSimpleCSRVectorConverter();
  virtual ~SYCLtoSimpleCSRVectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::sycl>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SYCLtoSimpleCSRVectorConverter::SYCLtoSimpleCSRVectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void SYCLtoSimpleCSRVectorConverter::convert(
const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SYCLVector<double>& sycl_v =
  cast<SYCLVector<double>>(sourceImpl, sourceBackend());
  SimpleCSRVector<double>& csr_v =
  cast<SimpleCSRVector<double>>(targetImpl, targetBackend());

  alien_debug([&] { cout() << "Converting SYCLVector: " << &sycl_v << " to SimpleCSRVector " << &csr_v; });
  sycl_v.copyValuesTo(csr_v.values().size(), csr_v.getDataPtr());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SYCLtoSimpleCSRVectorConverter);
