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
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <alien/petsc/backend.h>

class SimpleCSR_to_Petsc_VectorConverter : public Alien::IVectorConverter
{
 public:
  SimpleCSR_to_Petsc_VectorConverter() {}

  virtual ~SimpleCSR_to_Petsc_VectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const { return Alien::AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name(); }

  Alien::BackEndId targetBackend() const { return Alien::AlgebraTraits<Alien::BackEnd::tag::petsc>::name(); }

  void convert(const Alien::IVectorImpl* sourceImpl, Alien::IVectorImpl* targetImpl) const;
};

void SimpleCSR_to_Petsc_VectorConverter::convert(const Alien::IVectorImpl* sourceImpl,
                                                 Alien::IVectorImpl* targetImpl) const
{
  const auto& v = cast<Alien::SimpleCSRVector<Arccore::Real>>(sourceImpl, sourceBackend());
  auto& v2 = cast<Alien::PETSc::Vector>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting Alien::SimpleCSRVector: " << &v << " to PETSc::Vector " << &v2;
  });

  auto block_size = 1;
  const auto* block = v2.block();
  if (v2.block())
    block_size *= block->size();
  else if (v2.vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Not implemented yet");

  const auto localOffset = v2.distribution().offset();
  const auto localSize = v2.distribution().localSize();
  const auto ilower = localOffset * block_size;
  const auto iupper = ilower + localSize * block_size - 1;

  alien_debug([&] {
    cout() << "Vector range : "
           << "[" << ilower << ":" << iupper << "]";
  });

  auto values = v.values();

  v2.setValues(values);

  v2.assemble();
}

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Petsc_VectorConverter);
