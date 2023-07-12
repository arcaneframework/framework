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

#pragma once

#include <memory>
#include <alien/core/impl/IVectorImpl.h>

//#include <petscvec.h>
#include <ginkgo/core/matrix/dense.hpp>

namespace Alien::Ginkgo
{

class Vector : public IVectorImpl
, public gko::matrix::Dense<double>
{
 public:
  explicit Vector(const MultiVectorImpl* multi_impl);

  ~Vector() override;

 public:
  void setProfile(int ilower, int iupper);

  void setValues(Arccore::ConstArrayView<double> values);

  void getValues(Arccore::ArrayView<double> values) const;

  void assemble();

  // version using raw pointers
  gko::matrix::Dense<double> const* internal() const
  {
    return this;
  }

  gko::matrix::Dense<double>* internal()
  {
    return this;
  }

 private:
  // Vec m_vec;
  // MPI_Comm m_comm;
  gko::matrix_assembly_data<double> data;
  //Arccore::UniqueArray<Arccore::Integer> m_rows;
};

} // namespace Alien::Ginkgo
