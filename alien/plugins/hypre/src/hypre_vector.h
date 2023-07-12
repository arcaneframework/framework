/*
 * Copyright 2020-2021 IFPEN-CEA
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

#include <alien/core/impl/IVectorImpl.h>

#include <HYPRE_IJ_mv.h>

namespace Alien::Hypre
{
class VectorInternal;

class Vector : public IVectorImpl
{
 public:
  explicit Vector(const MultiVectorImpl* multi_impl);

  ~Vector() override;

  void setProfile(int ilower, int iupper);

  void setValues(Arccore::ConstArrayView<double> values);

  void getValues(Arccore::ArrayView<double> values) const;

  void assemble();

  HYPRE_IJVector internal() { return m_hypre; }

  HYPRE_IJVector internal() const { return m_hypre; }

 private:
  HYPRE_IJVector m_hypre = nullptr;
  MPI_Comm m_comm;

  Arccore::UniqueArray<Arccore::Integer> m_rows;
};

} // namespace Alien::Hypre
