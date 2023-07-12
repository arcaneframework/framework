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

#include "../vector.h"

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>

#include <alien/ginkgo/backend.h>

//SimpleCSR_to_Ginkgo_MatrixConverter
class Ginkgo_to_SimpleCSR_VectorConverter : public Alien::IVectorConverter
{
 public:
  Ginkgo_to_SimpleCSR_VectorConverter() = default;

  ~Ginkgo_to_SimpleCSR_VectorConverter() override = default;

 public:
  Alien::BackEndId sourceBackend() const override
  {
    return Alien::AlgebraTraits<Alien::BackEnd::tag::ginkgo>::name();
  }

  Alien::BackEndId targetBackend() const override
  {
    return Alien::AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name();
  }

  void convert(const Alien::IVectorImpl* sourceImpl,
               Alien::IVectorImpl* targetImpl) const override;
};

void Ginkgo_to_SimpleCSR_VectorConverter::convert(
const Alien::IVectorImpl* sourceImpl,
Alien::IVectorImpl* targetImpl) const
{
  const auto& v = cast<Alien::Ginkgo::Vector>(sourceImpl, sourceBackend());
  auto& v2 = cast<Alien::SimpleCSRVector<Arccore::Real>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting Ginkgo::Vector: " << &v
           << " to Alien::SimpleCSRVector " << &v2;
  });

  auto values = v2.values(); // array view
  v.getValues(values); // write into the array view
}

REGISTER_VECTOR_CONVERTER(Ginkgo_to_SimpleCSR_VectorConverter);
