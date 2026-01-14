// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <iostream>

#include <alien/utils/Precomp.h>
#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <alien/kernels/hypre/data_structure/HypreMatrix.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
#include <alien/distribution/MatrixDistribution.h>

#include <arccore/collections/Array2.h>

#include "SimpleCSR_to_Hypre_MatrixConverter.h"

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SimpleCSR_to_Hypre_MatrixConverter::SimpleCSR_to_Hypre_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_Hypre_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SimpleCSRMatrix<Arccore::Real>& v =
      cast<SimpleCSRMatrix<Arccore::Real>>(sourceImpl, sourceBackend());
  auto& v2 = cast<HypreMatrix>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRMatrix: " << &v << " to HypreMatrix " << &v2;
  });
  if (targetImpl->block())
    _buildBlock(v, v2);
  else if (targetImpl->vblock())
    throw Arccore::FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

void
SimpleCSR_to_Hypre_MatrixConverter::convert(
    const SimpleCSRMatrix<Arccore::Real>& source, HypreMatrix& target) const
{
  alien_debug([&] {
    cout() << "Converting SimpleCSRMatrix: " << &source << " to HypreMatrix " << &target;
  });
  if (target.block())
    _buildBlock(source, target);
  else if (target.vblock())
    throw Arccore::FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(source, target);

}
/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_Hypre_MatrixConverter::_build(
    const SimpleCSRMatrix<Arccore::Real>& sourceImpl, HypreMatrix& targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer localOffset = dist.rowOffset();
  const SimpleCSRMatrix<Arccore::Real>::MatrixInternal& matrixInternal =
      *sourceImpl.internal();

  Arccore::Integer data_count = 0;
  Arccore::Integer pos = 0;
  Arccore::Integer max_line_size = 0;
  Arccore::UniqueArray<int> sizes(localSize);
  Arccore::UniqueArray<int> row_uids(localSize);
  for (Arccore::Integer row = 0; row < localSize; ++row) {
    data_count += profile.getRowSize(row);
    sizes[pos] = profile.getRowSize(row);
    row_uids[pos] = localOffset + row;
    max_line_size = std::max(max_line_size, profile.getRowSize(row));
    ++pos;
  }

  int ilower = localOffset;
  int iupper = localOffset + localSize - 1;
  int jlower = ilower;
  int jupper = iupper;

  alien_debug([&] {
    cout() << "Matrix range : "
           << "[" << ilower << ":" << iupper << "]"
           << "x"
           << "[" << jlower << ":" << jupper << "]";
  });
  {

    if (not targetImpl.initMatrix(ilower, iupper, jlower, jupper, sizes)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre Initialisation failed");
    }

    if(targetImpl.getMemoryType() == BackEnd::Memory::Host )
    {
      alien_info([&] {
          cout() << "Copy values From Host to Host";
        });
      // Buffer de construction
      Arccore::UniqueArray<double> values(std::max(localSize, max_line_size));
      Arccore::UniqueArray<int>& indices = sizes; // réutilisation du buffer
      indices.resize(std::max(localSize, max_line_size));

      Arccore::ConstArrayView<Arccore::Real> m_values = matrixInternal.getValues();
      Arccore::ConstArrayView<Arccore::Integer> cols = profile.getCols();
      Arccore::Integer icount = 0;
      for (Arccore::Integer irow = 0; irow < localSize; ++irow) {
        int row = localOffset + irow;
        int ncols = profile.getRowSize(irow);
        Arccore::Integer jpos = 0;
        for (Arccore::Integer k = 0; k < ncols; ++k) {
          indices[jpos] = cols[icount];
          values[jpos] = m_values[icount];
          ++jpos;
          ++icount;
        }

        const bool success =
            targetImpl.setMatrixValues(1, &row, &ncols, indices.data(), values.data());

        if (not success) {
          throw Arccore::FatalErrorException(A_FUNCINFO,
              Arccore::String::format("Cannot set Hypre Matrix Values for row {0}", row));
        }
      }
    }
    else
    {
      alien_info([&] {
      cout() << "Copy values From Host to Device";
      });

      auto values = matrixInternal.getValues();
      auto cols = profile.getCols();
      const bool success = targetImpl.setMatrixValuesFrom(localSize,
                                                          data_count,
                                                          row_uids.data(),
                                                          sizes.data(),
                                                          cols.data(),
                                                          values.data(),
                                                          BackEnd::Memory::Host) ;
      if (not success) {
        throw Arccore::FatalErrorException(A_FUNCINFO,"Cannot set Hypre Matrix Values from Host to Device");
      }

    }
  }
  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre assembling failed");
  }
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_Hypre_MatrixConverter::_buildBlock(
    const SimpleCSRMatrix<Arccore::Real>& sourceImpl, HypreMatrix& targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer block_size = targetImpl.block()->size();
  const Arccore::Integer localOffset = dist.rowOffset();
  const SimpleCSRMatrix<Arccore::Real>::MatrixInternal& matrixInternal =
      *sourceImpl.internal();

  Arccore::Integer max_line_size = 0;
  Arccore::Integer data_count = 0;
  Arccore::Integer pos = 0;
  Arccore::UniqueArray<int> sizes(localSize * block_size);
  for (Arccore::Integer row = 0; row < localSize; ++row) {
    Arccore::Integer row_size = profile.getRowSize(row) * block_size;
    for (Arccore::Integer ieq = 0; ieq < block_size; ++ieq) {
      data_count += row_size;
      sizes[pos] = row_size;
      ++pos;
    }
    max_line_size = std::max(max_line_size, row_size);
  }

  int ilower = localOffset * block_size;
  int iupper = (localOffset + localSize) * block_size - 1;
  int jlower = ilower;
  int jupper = iupper;

  alien_debug([&] {
    cout() << "Matrix range : "
           << "[" << ilower << ":" << iupper << "]"
           << "x"
           << "[" << jlower << ":" << jupper << "]";
  });

  // est ce qu'on reconstruit la matrice  ?
  {
    if (not targetImpl.initMatrix(ilower, iupper, jlower, jupper, sizes)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre Initialisation failed");
    }

    // Buffer de construction
    Arccore::UniqueArray2<Arccore::Real> values(block_size, max_line_size);
    Arccore::UniqueArray<int>& indices = sizes; // réutilisation du buffer
    indices.resize(max_line_size);

    Arccore::ConstArrayView<Arccore::Integer> cols = profile.getCols();
    Arccore::ConstArrayView<Arccore::Real> m_values = matrixInternal.getValues();
    Arccore::Integer col_count = 0;
    Arccore::Integer mat_count = 0;
    for (Arccore::Integer irow = 0; irow < localSize; ++irow) {
      int row = localOffset + irow;
      int ncols = profile.getRowSize(irow);
      Arccore::Integer jcol = 0;
      for (Arccore::Integer k = 0; k < ncols; ++k)
        for (Arccore::Integer j = 0; j < block_size; ++j)
          indices[jcol++] = cols[col_count + k] * block_size + j;
      for (Arccore::Integer k = 0; k < ncols; ++k) {
        const Arccore::Integer kk = k * block_size * block_size;
        for (Arccore::Integer i = 0; i < block_size; ++i)
          for (Arccore::Integer j = 0; j < block_size; ++j)
            values[i][k * block_size + j] = m_values[mat_count + kk + i * block_size + j];
      }
      col_count += ncols;
      mat_count += ncols * block_size * block_size;

      for (Arccore::Integer i = 0; i < block_size; ++i) {
        Arccore::Integer rows = row * block_size + i;
        Arccore::Integer num_cols = ncols * block_size;
        const bool success = targetImpl.setMatrixValues(
            1, &rows, &num_cols, indices.data(), values[i].data());

        if (not success) {
          throw Arccore::FatalErrorException(A_FUNCINFO,
              Arccore::String::format("Cannot set Hypre Matrix Values for row {0}", row));
        }
      }
    }
  }

  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre assembling failed");
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_Hypre_MatrixConverter);
