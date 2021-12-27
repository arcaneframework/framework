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

class SimpleCSRtoSYCLVectorConverter : public IVectorConverter
{
 public:
  SimpleCSRtoSYCLVectorConverter();
  virtual ~SimpleCSRtoSYCLVectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::sycl>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSRtoSYCLVectorConverter::SimpleCSRtoSYCLVectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void SimpleCSRtoSYCLVectorConverter::convert(
const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SimpleCSRVector<double>& v =
  cast<SimpleCSRVector<double>>(sourceImpl, sourceBackend());
  SYCLVector<double>& v2 =
  cast<SYCLVector<double>>(targetImpl, targetBackend());

  alien_debug(
  [&] { cout() << "Converting SimpleCSRVector: " << &v << " to SYCLVector " << &v2; });

  ConstArrayView<Real> values = v.values();
  v2.setValues(v.scalarizedLocalSize(), values.data());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SimpleCSRtoSYCLVectorConverter);
