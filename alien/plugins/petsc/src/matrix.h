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

#include <alien/core/impl/IMatrixImpl.h>

#include <petscmat.h>

namespace Alien::PETSc
{
class Matrix : public IMatrixImpl
{
 public:
  explicit Matrix(const MultiMatrixImpl* multi_impl);

  ~Matrix() override;

 public:
  void setProfile(int ilower, int iupper, int jlower, int jupper,
                  [[maybe_unused]] Arccore::ConstArrayView<int> row_sizes);

  void setRowValues(int rows,
                    Arccore::ConstArrayView<int> cols,
                    Arccore::ConstArrayView<double> values);

  void assemble();

  Mat internal() const { return m_mat; }

 private:
  Mat m_mat;
  MPI_Comm m_comm{};
};

} // namespace Alien::PETSc
