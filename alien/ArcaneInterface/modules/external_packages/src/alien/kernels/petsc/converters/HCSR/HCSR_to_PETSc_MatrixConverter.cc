// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <alien/kernels/petsc/PETScBackEnd.h>
#include <alien/kernels/petsc/data_structure/PETScMatrix.h>
#include <alien/kernels/petsc/data_structure/PETScInternal.h>

#include <alien/kernels/sycl/data/HCSRMatrix.h>
#include <alien/core/block/ComputeBlockOffsets.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>

#include <alien/kernels/petsc/data_structure/PETScMatrix.h>
#include <alien/kernels/petsc/PETScBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class HCSR_to_PETSc_MatrixConverter : public IMatrixConverter
{
 public:
  HCSR_to_PETSc_MatrixConverter();
  virtual ~HCSR_to_PETSc_MatrixConverter() {}

 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hcsr>::name();
  }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::petsc>::name(); }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void _build(
      const HCSRMatrix<Arccore::Real>& sourceImpl, PETScMatrix& targetImpl) const;
  void _buildBlock(
      const HCSRMatrix<Arccore::Real>& sourceImpl, PETScMatrix& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

HCSR_to_PETSc_MatrixConverter::HCSR_to_PETSc_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
void
HCSR_to_PETSc_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const HCSRMatrix<Real>& v =
  cast<HCSRMatrix<Real>>(sourceImpl, sourceBackend());
  auto& v2 = cast<PETScMatrix>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting HCSRMatrix: " << &v << " to PETScMatrix " << &v2;
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
HCSR_to_PETSc_MatrixConverter::_build(
    const HCSRMatrix<Arccore::Real>& sourceImpl, PETScMatrix& targetImpl) const
{
  typedef Arccore::Real ValueType ;
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer localOffset = dist.rowOffset();
  const Arccore::Integer globalSize = dist.globalRowSize();
  const bool isParallel = dist.isParallel();

  {
    auto nnz = profile.getNnz();
    auto ghost_size = sourceImpl.getGhostSize() ;
    auto ndofs = localSize + ghost_size ;
    alien_debug([&] {
      cout() << "Build from HCSR: nrows = " << localSize<< " nnz = "<< nnz<< " ndofs = "<< ndofs;
    });

    int* dof_uids_d = nullptr ;
    int* rows_d = nullptr;
    int* cols_d = nullptr;
    ValueType* values_d = nullptr ;
    sourceImpl.initCOODevicePointers(&dof_uids_d, &rows_d, &cols_d, &values_d) ;
    alien_debug([&] {
      cout() << "Build from HCSR: initCOODevicePointers OK " ;
    });
    if (not targetImpl.initMatrix(localSize,
                                  localOffset,
                                  globalSize,
                                  1,
                                  ndofs,
                                  dof_uids_d,
                                  nnz,
                                  rows_d,
                                  cols_d,
                                  isParallel))
    {
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Initialisation failed");
    }

    alien_debug([&] {
      cout() << "Build from HCSR: initMatrix OK " ;
    });

    if(not targetImpl.setMatrixValuesFromCSR(values_d))
    {
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc setMatrixValue failed");
    }
    alien_debug([&] {
      cout() << "Build from HCSR: setMatrixValues OK " ;
    });

    sourceImpl.freeDevicePointers(dof_uids_d, rows_d, cols_d, values_d) ;
    alien_debug([&] {
      cout() << "Build from HCSR: freeDevicePointers OK " ;
    });
  }

  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc assembling failed");
  }
  alien_debug([&] {
    cout() << "Build from HCSR: assemble OK " ;
  });
}

/*---------------------------------------------------------------------------*/

void
HCSR_to_PETSc_MatrixConverter::_buildBlock(
    const HCSRMatrix<Arccore::Real>& sourceImpl, PETScMatrix& targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer block_size = targetImpl.block()->size();
  const Arccore::Integer localOffset = dist.rowOffset();
  const HCSRMatrix<Arccore::Real>::MatrixInternal& matrixInternal =
      *sourceImpl.internal();
  // est ce qu'on reconstruit la matrice  ?
  {
    /*
    if (not targetImpl.initMatrix(ilower, iupper, jlower, jupper, sizes)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Initialisation failed");
    }

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
              Arccore::String::format("Cannot set PETSc Matrix Values for row {0}", row));
        }
      }
    }*/
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Block Matrix set Values from Device memory not yet implemented");
  }

  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc assembling failed");
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(HCSR_to_PETSc_MatrixConverter);
