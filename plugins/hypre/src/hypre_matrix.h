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

#include <HYPRE_IJ_mv.h>

namespace Alien::Hypre
{
class Matrix : public IMatrixImpl
{
 public:
  Matrix(const MultiMatrixImpl* multi_impl);

  virtual ~Matrix();

 public:
  void setProfile(int ilower, int iupper,
                  int jlower, int jupper,
                  Arccore::ConstArrayView<int> row_sizes);

  void setRowValues(int rows,
                    Arccore::ConstArrayView<int> cols,
                    Arccore::ConstArrayView<double> values);

  void assemble();

  HYPRE_IJMatrix internal() const { return m_hypre; }

 private:
  HYPRE_IJMatrix m_hypre;
  MPI_Comm m_comm;
};

} // namespace Alien::Hypre
