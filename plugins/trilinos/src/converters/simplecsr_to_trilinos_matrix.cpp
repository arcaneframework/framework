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

#include "../trilinos_matrix.h"

#include <arccore/collections/Array2.h>

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <alien/trilinos/backend.h>

class SimpleCSR_to_Trilinos_MatrixConverter : public Alien::IMatrixConverter
{
 public:
  SimpleCSR_to_Trilinos_MatrixConverter() = default;

  ~SimpleCSR_to_Trilinos_MatrixConverter() final = default;

  BackEndId sourceBackend() const final
  {
    return Alien::AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name();
  }

  BackEndId targetBackend() const final { return Alien::AlgebraTraits<Alien::BackEnd::tag::trilinos>::name(); }

  void convert(const Alien::IMatrixImpl* sourceImpl, Alien::IMatrixImpl* targetImpl) const final;

  void _build(const Alien::SimpleCSRMatrix<Arccore::Real>& sourceImpl, Alien::Trilinos::Matrix& targetImpl) const;

  void _buildBlock(const Alien::SimpleCSRMatrix<Arccore::Real>& sourceImpl, Alien::Trilinos::Matrix& targetImpl) const;
};

void SimpleCSR_to_Trilinos_MatrixConverter::convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const auto& v = cast<Alien::SimpleCSRMatrix<Arccore::Real>>(sourceImpl, sourceBackend());
  auto& v2 = cast<Alien::Trilinos::Matrix>(targetImpl, targetBackend());

  if (targetImpl->block())
    _buildBlock(v, v2);
  else if (targetImpl->vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

void SimpleCSR_to_Trilinos_MatrixConverter::_build(const Alien::SimpleCSRMatrix<Arccore::Real>& sourceImpl,
                                                   Alien::Trilinos::Matrix& targetImpl) const
{
  const auto& dist = sourceImpl.distribution();
  const auto& profile = sourceImpl.getCSRProfile();
  const auto localSize = profile.getNRow();
  const auto localOffset = dist.rowOffset();

  int globalSize = dist.rowSpace().size();

  auto sizes = Arccore::UniqueArray<int>(localSize);
  for (auto row = 0; row < localSize; ++row) {
    sizes[row] = profile.getRowSize(row);
  }

  targetImpl.setProfile(localSize, globalSize, sizes);

  auto values = sourceImpl.internal().getValues();
  auto cols = profile.getCols();
  for (auto irow = 0; irow < localSize; ++irow) {
    const auto row = localOffset + irow;
    const auto ncols = profile.getRowSize(irow);
    const auto col_offset = profile.getRowOffset()[irow];
    targetImpl.setRowValues(row, cols.subConstView(col_offset, ncols), values.subConstView(col_offset, ncols));
  }

  targetImpl.assemble();
}

void SimpleCSR_to_Trilinos_MatrixConverter::_buildBlock(const Alien::SimpleCSRMatrix<Arccore::Real>& sourceImpl,
                                                        Alien::Trilinos::Matrix& targetImpl) const
{
  throw Arccore::NotImplementedException(A_FUNCINFO, "SimpleCSR_to_Trilinos_MatrixConverter::_buildBlock not implemented");
}

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_Trilinos_MatrixConverter);
