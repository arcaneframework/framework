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

#include <alien/core/backend/VectorConverterRegisterer.h>

#include <alien/kernels/composite/CompositeVector.h>

#include <alien/handlers/scalar/BaseVectorReader.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

class Composite_to_SimpleCSR_VectorConverter : public Alien::IVectorConverter
{
 public:
  Composite_to_SimpleCSR_VectorConverter() {}

  virtual ~Composite_to_SimpleCSR_VectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const
  {
    return backendId<Alien::BackEnd::tag::composite>();
  }
  Alien::BackEndId targetBackend() const
  {
    return backendId<Alien::BackEnd::tag::simplecsr>();
  }

  void convert(
  const Alien::IVectorImpl* sourceImpl, Alien::IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Composite_to_SimpleCSR_VectorConverter::convert(
const Alien::IVectorImpl* sourceImpl, Alien::IVectorImpl* targetImpl) const
{
  const auto& v = cast<Alien::CompositeKernel::Vector>(sourceImpl, sourceBackend());
  auto& v2 = cast<Alien::SimpleCSRVector<Real>>(targetImpl, targetBackend());
  auto values = v2.fullValues();

  Integer index = 0;
  for (Integer i = 0; i < v.size(); ++i) {
    auto& c = v[i];
    Alien::Common::LocalVectorReader reader(c);
    for (Integer j = 0; j < c.impl()->space().size(); j++) {
      values[j + index] = reader[j];
    }
    index += c.impl()->space().size();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(Composite_to_SimpleCSR_VectorConverter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
