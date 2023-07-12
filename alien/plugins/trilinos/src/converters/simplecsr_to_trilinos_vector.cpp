/*
 * Copyright 2022 IFPEN-CEA
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

#include "../trilinos_vector.h"

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <alien/trilinos/backend.h>

class SimpleCSR_to_Trilinos_VectorConverter : public Alien::IVectorConverter
{
 public:
  SimpleCSR_to_Trilinos_VectorConverter() = default;

  ~SimpleCSR_to_Trilinos_VectorConverter() final = default;

  Alien::BackEndId sourceBackend() const final { return Alien::AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name(); }

  Alien::BackEndId targetBackend() const final { return Alien::AlgebraTraits<Alien::BackEnd::tag::trilinos>::name(); }

  void convert(const Alien::IVectorImpl* sourceImpl, Alien::IVectorImpl* targetImpl) const final;
};

void SimpleCSR_to_Trilinos_VectorConverter::convert(const Alien::IVectorImpl* sourceImpl,
                                                    Alien::IVectorImpl* targetImpl) const
{
  const auto& v = cast<Alien::SimpleCSRVector<Arccore::Real>>(sourceImpl, sourceBackend());
  auto& v2 = cast<Alien::Trilinos::Vector>(targetImpl, targetBackend());

  if (v2.vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Not implemented yet");

  // get data from source (Alien)
  auto values = v.values();

  // write into dest (Trilinos)
  v2.setValues(values);
}

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Trilinos_VectorConverter);
