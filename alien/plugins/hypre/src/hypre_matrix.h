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

#include <alien/core/impl/IMatrixImpl.h>

#include <HYPRE_IJ_mv.h>

namespace Alien::Hypre
{
class Matrix : public IMatrixImpl
{
 public:
  explicit Matrix(const MultiMatrixImpl* multi_impl);

  ~Matrix() override;

  void setProfile(Arccore::ConstArrayView<int> row_sizes);

  void setRowValues(int row,
                    Arccore::ConstArrayView<int> cols,
                    Arccore::ConstArrayView<double> values);

  //! Fill several partial rows at the same time.
  //! Function strongly mimic `HYPRE_IJMatrixSetValues` semantic.
  //!
  //! \param rows     array of row ids
  //! \param ncols    array of numbers of columns for each row id
  //! \param cols     array of column ids
  //! \param values   array of values
  //!
  //! `rows` and `ncols` should have the same size.
  //! `cols` and `values` should have the same size.
  //! For Hypre to use OpenMP threads for set values, rows values must be unique.
  void setRowsValues(Arccore::ConstArrayView<int> rows,
                     Arccore::ArrayView<int> ncols,
                     Arccore::ConstArrayView<int> cols,
                     Arccore::ConstArrayView<double> values);

  void assemble();

  HYPRE_IJMatrix internal() const { return m_hypre; }

 private:
  void init();

  HYPRE_IJMatrix m_hypre = nullptr;
  MPI_Comm m_comm;
};

} // namespace Alien::Hypre
