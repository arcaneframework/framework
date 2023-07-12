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

#include "../hypre_matrix.h"

#include <arccore/collections/Array2.h>

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <alien/hypre/backend.h>

namespace
{
class SimpleCSR_to_Hypre_MatrixConverter final : public Alien::IMatrixConverter
{
 public:
  SimpleCSR_to_Hypre_MatrixConverter() = default;

  ~SimpleCSR_to_Hypre_MatrixConverter() override = default;

  BackEndId sourceBackend() const override
  {
    return Alien::AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name();
  }

  BackEndId targetBackend() const override { return Alien::AlgebraTraits<Alien::BackEnd::tag::hypre>::name(); }

  void convert(const Alien::IMatrixImpl* sourceImpl, Alien::IMatrixImpl* targetImpl) const override;

  void _build(const Alien::SimpleCSRMatrix<Arccore::Real>& sourceImpl, Alien::Hypre::Matrix& targetImpl) const;

  void _buildBlock(const Alien::SimpleCSRMatrix<Arccore::Real>& sourceImpl, Alien::Hypre::Matrix& targetImpl) const;
};

void SimpleCSR_to_Hypre_MatrixConverter::convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const auto& v = cast<Alien::SimpleCSRMatrix<Arccore::Real>>(sourceImpl, sourceBackend());
  auto& v2 = cast<Alien::Hypre::Matrix>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting Alien::SimpleCSRMatrix: " << &v << " to Hypre::Matrix " << &v2;
  });

  if (targetImpl->block())
    _buildBlock(v, v2);
  else if (targetImpl->vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

void SimpleCSR_to_Hypre_MatrixConverter::_build(const Alien::SimpleCSRMatrix<Arccore::Real>& sourceImpl,
                                                Alien::Hypre::Matrix& targetImpl) const
{
  const auto& dist = sourceImpl.distribution();
  const auto& profile = sourceImpl.getCSRProfile();
  const auto localSize = profile.getNRow();
  const auto localOffset = dist.rowOffset();

  // Number of columns for each row
  auto ncols = Arccore::UniqueArray<int>(localSize);
  // Global Id for each row
  auto rows = Arccore::UniqueArray<int>(localSize);
  for (auto irow = 0; irow < localSize; ++irow) {
    ncols[irow] = profile.getRowSize(irow);
    rows[irow] = localOffset + irow;
  }

  targetImpl.setProfile(ncols);

  auto values = sourceImpl.internal().getValues();
  auto cols = profile.getCols();

  // understand why values and cols can have different sizes !
  targetImpl.setRowsValues(rows, ncols, cols, values.subConstView(0, cols.size()));

  targetImpl.assemble();
}

void SimpleCSR_to_Hypre_MatrixConverter::_buildBlock(const Alien::SimpleCSRMatrix<Arccore::Real>& sourceImpl,
                                                     Alien::Hypre::Matrix& targetImpl) const
{
  const auto& dist = sourceImpl.distribution();
  const auto& profile = sourceImpl.getCSRProfile();
  const auto localSize = profile.getNRow();
  const auto block_size = targetImpl.block()->size();
  const auto localOffset = dist.rowOffset();
  const auto& matrixInternal = sourceImpl.internal();

  auto max_line_size = localSize * block_size;
  auto pos = 0;
  Arccore::UniqueArray<int> sizes(localSize * block_size);
  for (auto row = 0; row < localSize; ++row) {
    auto row_size = profile.getRowSize(row) * block_size;
    for (auto ieq = 0; ieq < block_size; ++ieq) {
      sizes[pos] = row_size;
      ++pos;
    }
    max_line_size = std::max(max_line_size, row_size);
  }

  auto ilower = localOffset * block_size;
  auto iupper = (localOffset + localSize) * block_size - 1;
  auto jlower = ilower;
  auto jupper = iupper;

  alien_debug([&] {
    cout() << "Matrix range : "
           << "[" << ilower << ":" << iupper << "]"
           << "x"
           << "[" << jlower << ":" << jupper << "]";
  });

  // Buffer de construction
  Arccore::UniqueArray2<Arccore::Real> values;
  values.resize(block_size, max_line_size);
  Arccore::UniqueArray<int>& indices = sizes; // réutilisation du buffer
  indices.resize(std::max(max_line_size, localSize * block_size));

  targetImpl.setProfile(sizes);

  auto cols = profile.getCols();
  auto m_values = matrixInternal.getValues();
  auto col_count = 0;
  auto mat_count = 0;
  for (auto irow = 0; irow < localSize; ++irow) {
    int row = localOffset + irow;
    int ncols = profile.getRowSize(irow);
    auto jcol = 0;
    for (auto k = 0; k < ncols; ++k)
      for (auto j = 0; j < block_size; ++j)
        indices[jcol++] = cols[col_count + k] * block_size + j;
    for (auto k = 0; k < ncols; ++k) {
      const auto kk = k * block_size * block_size;
      for (auto i = 0; i < block_size; ++i)
        for (auto j = 0; j < block_size; ++j)
          values[i][k * block_size + j] = m_values[mat_count + kk + i * block_size + j];
    }
    col_count += ncols;
    mat_count += ncols * block_size * block_size;

    for (auto i = 0; i < block_size; ++i) {
      auto rows = row * block_size + i;
      auto num_cols = ncols * block_size;
      targetImpl.setRowValues(rows, indices.subConstView(0, num_cols), values[i].subConstView(0, num_cols));
    }
  }

  targetImpl.assemble();
}
} // namespace

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_Hypre_MatrixConverter);
