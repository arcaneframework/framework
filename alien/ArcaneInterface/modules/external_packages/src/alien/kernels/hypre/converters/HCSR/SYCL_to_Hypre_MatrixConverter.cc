// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/data_structure/HypreInternal.h>

#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/core/block/ComputeBlockOffsets.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>

#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/HypreBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

#include "SYCL_to_Hypre_MatrixConverter.h"


/*---------------------------------------------------------------------------*/

SYCL_to_Hypre_MatrixConverter::SYCL_to_Hypre_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
void
SYCL_to_Hypre_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SYCLBEllPackMatrix<Real>& source =
      cast<SYCLBEllPackMatrix<Real>>(sourceImpl, sourceBackend());
  auto& target = cast<HypreMatrix>(targetImpl, targetBackend());
}

void
SYCL_to_Hypre_MatrixConverter::convert(
    const SYCLBEllPackMatrix<Real>& source, HypreMatrix& target) const
{
  alien_debug([&] {
    cout() << "Converting SYCLBEllPackMatrix: " << &source << " to HypreMatrix " << &target;
  });
  if (source.blockSize()>1)
    _buildBlock(source, target);
  else if (target.vblock())
    throw Arccore::FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(source, target);
}

/*---------------------------------------------------------------------------*/

void
SYCL_to_Hypre_MatrixConverter::_build(
    const SYCLBEllPackMatrix<Arccore::Real>& sourceImpl, HypreMatrix& targetImpl) const
{
  typedef Arccore::Real ValueType ;
  const MatrixDistribution& dist = sourceImpl.distribution();
  auto const& profile = sourceImpl.getProfile();
  auto localSize      = profile.getNRows();
  auto localOffset    = dist.rowOffset();
  auto nnz            = profile.getNnz();
  auto kcol           = profile.kcol() ;

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

    Arccore::Integer data_count = 0;
    Arccore::Integer pos = 0;
    Arccore::Integer max_line_size = 0;
    Arccore::UniqueArray<int> sizes(localSize);
    for (Arccore::Integer row = 0; row < localSize; ++row) {
      auto row_size = kcol[row+1] - kcol[row];
      data_count += row_size;
      sizes[pos] = row_size;
      max_line_size = std::max(max_line_size, row_size);
      ++pos;
    }
    assert(data_count==nnz) ;

    if (not targetImpl.initMatrix(ilower, iupper, jlower, jupper, sizes)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre Initialisation failed");
    }

    {
      //SYCLBEllPackMatrix<Arccore::Real>::HCSRView view =
      //    sourceImpl.hcsrView(BackEnd::Memory::Device,localSize,nnz) ;
      auto view = targetImpl.hcsrView(BackEnd::Memory::Device,localSize,nnz) ;
      sourceImpl.copyDevicePointers(localSize, nnz, view.m_rows, view.m_ncols, view.m_cols, view.m_values) ;
      if(not targetImpl.setMatrixValuesFrom(localSize,
                                            nnz,
                                            view.m_rows,
                                            view.m_ncols,
                                            view.m_cols,
                                            view.m_values,
                                            BackEnd::Memory::Device))
        throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre setMatrixValues failed");
    }
  }
  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre assembling failed");
  }
}

/*---------------------------------------------------------------------------*/

void
SYCL_to_Hypre_MatrixConverter::_buildBlock(
    const SYCLBEllPackMatrix<Arccore::Real>& sourceImpl, HypreMatrix& targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  auto const& profile = sourceImpl.getProfile();
  auto localSize      = profile.getNRows();
  auto block_size     = targetImpl.block()->size();
  auto localOffset    = dist.rowOffset();
  auto kcol           = profile.kcol() ;

  Arccore::Integer max_line_size = localSize * block_size;
  Arccore::Integer data_count = 0;
  Arccore::Integer pos = 0;
  Arccore::UniqueArray<int> sizes(localSize * block_size);
  for (Arccore::Integer row = 0; row < localSize; ++row) {
    auto row_size = (kcol[row+1] - kcol[row]) * block_size;
    for (Arccore::Integer ieq = 0; ieq < block_size; ++ieq)
    {
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

  // Buffer de construction
  Arccore::UniqueArray2<Arccore::Real> values;
  values.resize(block_size, max_line_size);
  // est ce qu'on reconstruit la matrice  ?
  {
    if (not targetImpl.initMatrix(ilower, iupper, jlower, jupper, sizes)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre Initialisation failed");
    }
    /*
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
    }*/
    throw Arccore::FatalErrorException(A_FUNCINFO, "HYPRE Block Matrix set Values from Device memory not yet implemented");
  }

  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre assembling failed");
  }
}


/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(SYCL_to_Hypre_MatrixConverter);
