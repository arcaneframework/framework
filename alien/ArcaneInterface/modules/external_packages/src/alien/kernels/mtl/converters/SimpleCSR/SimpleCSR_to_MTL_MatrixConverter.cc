// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/mtl/data_structure/MTLVector.h>
#include <alien/kernels/mtl/data_structure/MTLMatrix.h>
#include <alien/core/block/ComputeBlockOffsets.h>

#include <alien/kernels/mtl/MTLBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_MTL_MatrixConverter : public IMatrixConverter
{
 public:
  SimpleCSR_to_MTL_MatrixConverter();
  virtual ~SimpleCSR_to_MTL_MatrixConverter() {}

 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::mtl>::name(); }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void _build(
      const SimpleCSRMatrix<Arccore::Real>& sourceImpl, MTLMatrix& targetImpl) const;
  void _buildBlock(
      const SimpleCSRMatrix<Arccore::Real>& sourceImpl, MTLMatrix& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_MTL_MatrixConverter::SimpleCSR_to_MTL_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_MTL_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SimpleCSRMatrix<Arccore::Real>& v =
      cast<SimpleCSRMatrix<Arccore::Real>>(sourceImpl, sourceBackend());
  MTLMatrix& v2 = cast<MTLMatrix>(targetImpl, targetBackend());

  alien_debug(
      [&] { cout() << "Converting SimpleCSRMatrix: " << &v << " to MTLMatrix " << &v2; });

  if (targetImpl->block())
    _buildBlock(v, v2);
  else if (targetImpl->vblock())
    throw Arccore::FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

void
SimpleCSR_to_MTL_MatrixConverter::_build(
    const SimpleCSRMatrix<Arccore::Real>& sourceImpl, MTLMatrix& targetImpl) const
{
  const MatrixDistribution& dist = targetImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer localOffset = dist.rowOffset();
  const SimpleCSRMatrix<Arccore::Real>::MatrixInternal& matrixInternal =
      *sourceImpl.internal();
  const Arccore::Integer myRank = dist.parallelMng()->commRank();
  const Arccore::Integer nProc = dist.parallelMng()->commSize();

  std::vector<std::size_t> offsets(nProc + 1);
  for (Arccore::Integer i = 0; i < nProc; ++i)
    offsets[i] = dist.rowOffset(i);
  offsets[nProc] = dist.globalRowSize();

  // Buffer de construction
  Arccore::Integer max_line_size = 0;
  Arccore::UniqueArray<Arccore::Integer> rows(localSize);
  for (Arccore::Integer irow = 0; irow < localSize; ++irow) {
    rows[irow] = localOffset + irow;
    max_line_size = std::max(max_line_size, profile.getRowSize(irow));
  }

  {
    Arccore::ConstArrayView<Arccore::Integer> row_offset = profile.getRowOffset();
    Arccore::ConstArrayView<Arccore::Integer> cols = profile.getCols();
    Arccore::ConstArrayView<Arccore::Real> m_values = matrixInternal.getValues();
    if (not targetImpl.initMatrix(offsets, myRank, nProc)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "MTL4 Initialisation failed");
    }

    const bool success = targetImpl.setMatrixValues(max_line_size, localSize, rows.data(),
        row_offset.data(), cols.data(), m_values.data());
    if (not success) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Cannot set MTL4 Matrix Values");
    }
  }
}

void
SimpleCSR_to_MTL_MatrixConverter::_buildBlock(
    const SimpleCSRMatrix<Arccore::Real>& sourceImpl, MTLMatrix& targetImpl) const
{

  const MatrixDistribution& dist = targetImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer localOffset = dist.rowOffset();
  const SimpleCSRMatrix<Arccore::Real>::MatrixInternal& matrixInternal =
      *sourceImpl.internal();
  const Arccore::Integer myRank = dist.parallelMng()->commRank();
  const Arccore::Integer nProc = dist.parallelMng()->commSize();
  const Block& block = *targetImpl.block();
  const Arccore::Integer block_size = block.size();

  Arccore::Integer max_line_size = 0;
  for (Arccore::Integer row = 0; row < localSize; ++row) {
    Arccore::Integer row_size = profile.getRowSize(row) * block_size;
    max_line_size = std::max(max_line_size, row_size);
  }

  std::vector<std::size_t> offsets;
  computeBlockOffsets(sourceImpl.distribution(), block, offsets);

  {
    if (not targetImpl.initMatrix(offsets, myRank, nProc)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "MTL4 Initialisation failed");
    }

    Arccore::ConstArrayView<Arccore::Integer> row_offset = profile.getRowOffset();
    Arccore::ConstArrayView<Arccore::Integer> cols = profile.getCols();
    Arccore::ConstArrayView<Arccore::Real> m_values = matrixInternal.getValues();
    const bool success =
        targetImpl.setMatrixBlockValues(localOffset, max_line_size, block_size,
            block_size, localSize, row_offset.data(), cols.data(), m_values.data());
    if (not success) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Cannot set MTL4 Matrix Values");
    }
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_MTL_MatrixConverter);
