// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/data_structure/HypreInternal.h>

#include <alien/kernels/sycl/data/HCSRMatrix.h>
#include <alien/core/block/ComputeBlockOffsets.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>

#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/HypreBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class HCSR_to_Hypre_MatrixConverter : public IMatrixConverter
{
 public:
  HCSR_to_Hypre_MatrixConverter();
  virtual ~HCSR_to_Hypre_MatrixConverter() {}

 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hcsr>::name();
  }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::hypre>::name(); }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void _build(
      const HCSRMatrix<Arccore::Real>& sourceImpl, HypreMatrix& targetImpl) const;
  void _buildBlock(
      const HCSRMatrix<Arccore::Real>& sourceImpl, HypreMatrix& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

HCSR_to_Hypre_MatrixConverter::HCSR_to_Hypre_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
void
HCSR_to_Hypre_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const HCSRMatrix<Real>& v =
  cast<HCSRMatrix<Real>>(sourceImpl, sourceBackend());
  auto& v2 = cast<HypreMatrix>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting HCSRMatrix: " << &v << " to HypreMatrix " << &v2;
  });
  if (targetImpl->block())
    _buildBlock(v, v2);
  else if (targetImpl->vblock())
    throw Arccore::FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

/*---------------------------------------------------------------------------*/

void
HCSR_to_Hypre_MatrixConverter::_build(
    const HCSRMatrix<Arccore::Real>& sourceImpl, HypreMatrix& targetImpl) const
{
  typedef Arccore::Real ValueType ;
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer localOffset = dist.rowOffset();

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
      data_count += profile.getRowSize(row);
      sizes[pos] = profile.getRowSize(row);
      max_line_size = std::max(max_line_size, profile.getRowSize(row));
      ++pos;
    }

    if (not targetImpl.initMatrix(ilower, iupper, jlower, jupper, sizes)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre Initialisation failed");
    }

    {
      int* ncols_d = nullptr;
      int* rows_d = nullptr;
      int* cols_d = nullptr;
      ValueType* values_d = nullptr ;
      sourceImpl.initDevicePointers(&ncols_d, &rows_d, &cols_d, &values_d) ;
      const bool success = targetImpl.setMatrixValues(localSize, rows_d, ncols_d, cols_d, values_d) ;
      sourceImpl.freeDevicePointers(ncols_d, rows_d, cols_d, values_d) ;
    }
  }
  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre assembling failed");
  }
}

/*---------------------------------------------------------------------------*/

void
HCSR_to_Hypre_MatrixConverter::_buildBlock(
    const HCSRMatrix<Arccore::Real>& sourceImpl, HypreMatrix& targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer block_size = targetImpl.block()->size();
  const Arccore::Integer localOffset = dist.rowOffset();
  const HCSRMatrix<Arccore::Real>::MatrixInternal& matrixInternal =
      *sourceImpl.internal();

  Arccore::Integer max_line_size = localSize * block_size;
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

  // Buffer de construction
  Arccore::UniqueArray2<Arccore::Real> values;
  values.resize(block_size, max_line_size);
  Arccore::UniqueArray<int>& indices = sizes; // réutilisation du buffer
  indices.resize(std::max(max_line_size, localSize * block_size));
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
  }

  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre assembling failed");
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(HCSR_to_Hypre_MatrixConverter);
