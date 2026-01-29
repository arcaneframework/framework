// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <alien/kernels/petsc/data_structure/PETScMatrix.h>
#include <alien/kernels/petsc/data_structure/PETScVector.h>
#include <iostream>

#include <alien/kernels/petsc/PETScBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <arccore/collections/Array2.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_PETSc_MatrixConverter : public IMatrixConverter
{
 public:
  SimpleCSR_to_PETSc_MatrixConverter();
  virtual ~SimpleCSR_to_PETSc_MatrixConverter() {}
 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::petsc>::name(); }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void _build(
      const SimpleCSRMatrix<Arccore::Real>& sourceImpl, PETScMatrix& targetImpl) const;
  void _buildBlock(
      const SimpleCSRMatrix<Arccore::Real>& sourceImpl, PETScMatrix& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_PETSc_MatrixConverter::SimpleCSR_to_PETSc_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_PETSc_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SimpleCSRMatrix<Arccore::Real>& v =
      cast<SimpleCSRMatrix<Arccore::Real>>(sourceImpl, sourceBackend());
  PETScMatrix& v2 = cast<PETScMatrix>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRMatrix: " << &v << " to PETScMatrix " << &v2;
  });
  if (sourceImpl->block())
    _buildBlock(v, v2);
  else if (sourceImpl->vblock())
    throw Arccore::FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_PETSc_MatrixConverter::_build(
    const SimpleCSRMatrix<Arccore::Real>& sourceImpl, PETScMatrix& targetImpl) const
{
  const MatrixDistribution& dist = targetImpl.distribution();

  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer globalSize = dist.globalRowSize();
  const Arccore::Integer localOffset = dist.rowOffset();
  const bool isParallel = dist.isParallel();

  const SimpleCSRMatrix<Arccore::Real>::MatrixInternal& matrixInternal =
      *sourceImpl.internal();

  alien_info([&] {
    cout() << "Matrix range : [" << localOffset << ":" << localOffset + localSize<< "]";
  });

  auto exec_space = targetImpl.getExecSpace() ;
  if(exec_space == BackEnd::Exec::Host)
  {
    alien_info([&] {
      cout() << "Exec space : HOST";
    });
    Arccore::Integer max_line_size = localSize; // taille mini pour y mettre rhs
    Arccore::UniqueArray<Arccore::Integer> diag_sizes(localSize);
    Arccore::UniqueArray<Arccore::Integer> offdiag_sizes(localSize);

    Arccore::ConstArrayView<Arccore::Integer> cols = profile.getCols();
    Arccore::Integer const* row_cols_ptr = cols.data();

    for (Arccore::Integer irow = 0; irow < localSize; ++irow) {
      int ncols = profile.getRowSize(irow);
      Arccore::Integer diag_count = 0;
      Arccore::Integer offdiag_count = 0;
      for (Arccore::Integer i = 0; i < ncols; ++i) {
        if ((row_cols_ptr[i] < localOffset) || (row_cols_ptr[i] >= localOffset + localSize))
          ++offdiag_count;
        else
          ++diag_count;
      }
      diag_sizes[irow] = std::max(
          1, diag_count); // Toujours au moins une valeur pour la diagonale dixit PETSc
      offdiag_sizes[irow] = offdiag_count;
      max_line_size = std::max(max_line_size, diag_count + offdiag_count);
      row_cols_ptr += ncols;
    }

    Arccore::UniqueArray<Arccore::Real> values(max_line_size);
    Arccore::UniqueArray<Arccore::Integer> indices(max_line_size);

    {
      if (not targetImpl.initMatrix(localSize,
                                    localOffset,
                                    globalSize,
                                    1,
                                    diag_sizes,
                                    offdiag_sizes,
                                    isParallel))
      {
        throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Initialisation failed");
      }

      {
        Arccore::ConstArrayView<Arccore::Integer> cols = profile.getCols();
        Arccore::ConstArrayView<Arccore::Real> m_values = matrixInternal.getValues();
        Arccore::Integer icount = 0;
        for (Arccore::Integer irow = 0; irow < localSize; ++irow) {
          int row = localOffset + irow;
          int ncols = profile.getRowSize(irow);
          for (Arccore::Integer k = 0; k < ncols; ++k) {
            indices[k] = cols[icount];
            values[k] = m_values[icount];
            ++icount;
          }
          const bool success =
              targetImpl.setMatrixValues(row, ncols, indices.data(), values.data());

          if (not success) {
            throw Arccore::FatalErrorException(A_FUNCINFO,
                Arccore::String::format("Cannot set PETSc Matrix Values for row ", row));
          }
        }
      }
    }
  }
  else
  {
    alien_info([&] {
      cout() << "Exec space : DEVICE";
    });
    auto nnz = profile.getNnz();
    auto ghost_size = sourceImpl.getGhostSize() ;
    auto kcol = profile.kcol();
    auto cols = profile.cols();
    auto values = sourceImpl.data();

    auto ndofs = localSize + ghost_size ;
    std::vector<int> dof_uids(ndofs) ;
    std::vector<int> rows(nnz) ;
    for(int i=0;i<localSize;++i)
    {
      auto uid = localOffset + i ;
      dof_uids[i] = uid ;
      for(int k=kcol[i];k<kcol[i+1];++k)
        rows[k] = uid ;
    }
    if(isParallel)
    {
      auto const& ghost_uids = sourceImpl.getDistStructInfo().m_recv_info.m_uids ;
      for(int i=0;i<ghost_size;++i)
            dof_uids[localSize+i] = ghost_uids[i] ;
    }

    if (not targetImpl.initMatrix(localSize,
                                  localOffset,
                                  globalSize,
                                  1,
                                  ndofs,
                                  dof_uids.data(),
                                  nnz,
                                  rows.data(),
                                  (int*)cols,
                                  isParallel))
    {
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Initialisation failed");
    }

    if(not targetImpl.setMatrixValuesFromCSR(values))
    {
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc setMatrixValue failed");
    }

  }

  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc assembling failed");
  }
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_PETSc_MatrixConverter::_buildBlock(
    const SimpleCSRMatrix<Arccore::Real>& sourceImpl, PETScMatrix& targetImpl) const
{
  const MatrixDistribution& dist = targetImpl.distribution();
  const Block* block = targetImpl.block();

  const Arccore::Integer block_size = block->size();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = dist.localRowSize();
  const Arccore::Integer localScalarizedSize = localSize * block_size;
  const Arccore::Integer globalScalarizedSize = dist.globalRowSize() * block_size;
  const Arccore::Integer localOffset = dist.rowOffset();
  const Arccore::Integer localScalarizedOffset = dist.rowOffset() * block_size;
  const bool isParallel = dist.isParallel();

  const SimpleCSRMatrix<Arccore::Real>::MatrixInternal& matrixInternal =
      *sourceImpl.internal();

  Arccore::UniqueArray<Arccore::Integer> diag_sizes(localScalarizedSize);
  Arccore::UniqueArray<Arccore::Integer> offdiag_sizes(localScalarizedSize);

  Arccore::ConstArrayView<Arccore::Integer> cols = profile.getCols();
  Arccore::Integer const* row_cols_ptr = cols.data();

  //Arccore::Integer max_line_size = localScalarizedSize; // taille mini pour y mettre rhs
  Arccore::Integer max_line_size = 0 ;
  for (Arccore::Integer irow = 0, row = 0; irow < localSize; ++irow)
  {
    int ncols = matrixInternal.getRowSize(irow);
    Arccore::Integer diag_count = 0;
    Arccore::Integer offdiag_count = 0;
    for (Arccore::Integer i = 0; i < ncols; ++i) {
      if ((row_cols_ptr[i] < localOffset) || (row_cols_ptr[i] >= localOffset + localSize))
        ++offdiag_count;
      else
        ++diag_count;
    }

    // Toujours au moins une valeur pour la diagonale dixit PETSc
    Arccore::Integer diag_size = std::max(block_size,diag_count*block_size);
    Arccore::Integer offdiag_size = offdiag_count * block_size;
    for(Integer b=0;b<block_size;++b,++row)
    {
      diag_sizes[row] = diag_size;
      offdiag_sizes[row] = offdiag_size;
    }

    max_line_size = std::max(max_line_size, (diag_count + offdiag_count) * block_size);
    row_cols_ptr += ncols;
  }

  // std::cout << "diag_sizes: " << diag_sizes << "\n";
  Arccore::UniqueArray<Arccore::Integer> indices(max_line_size);

  {
    if (not targetImpl.initMatrix(localScalarizedSize,
                                  localScalarizedOffset,
                                  globalScalarizedSize,
                                  block_size,
                                  diag_sizes,
                                  offdiag_sizes,
                                  isParallel)) {
      throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Initialisation failed");
    }
#ifdef PETSC_HAVE_MATSETBLOCKSIZE
    {
      Arccore::UniqueArray<Arccore::Real> values;
      values.resize(block_size * max_line_size);
      Arccore::ConstArrayView<Arccore::Integer> cols = profile.getCols();
      Arccore::ConstArrayView<Arccore::Real> matrix_values = matrixInternal.getValues();
      Arccore::Integer col_count = 0;
      Arccore::Integer mat_count = 0;
      for (Arccore::Integer irow = 0; irow < localSize; ++irow) {
        int row = localOffset + irow;
        int ncols = profile.getRowSize(irow);
        for (Arccore::Integer k = 0; k < ncols; ++k)
          indices[k] = cols[col_count + k];

        for (Arccore::Integer k = 0; k < ncols; ++k)
        {
          const Arccore::Integer kk = k * block_size * block_size;
          for (Arccore::Integer i = 0; i < block_size; ++i)
            for (Arccore::Integer j = 0; j < block_size; ++j)
              values[i *ncols*block_size + k * block_size + j] =
                  matrix_values[mat_count + kk + i * block_size + j];
        }
        col_count += ncols;
        mat_count += ncols * block_size * block_size;

        const bool success = targetImpl.setMatrixBlockValues(row,
                                                             ncols,
                                                             indices.data(),
                                                             values.data());

        if (not success) {
          throw Arccore::FatalErrorException(A_FUNCINFO,
              Arccore::String::format("Cannot set PETSc Matrix Values for row ", row));
        }
      }
    }
#else
    {
      Arccore::UniqueArray2<Arccore::Real> values(block_size,max_line_size);
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
              values[i][k * block_size + j] =
                  m_values[mat_count + kk + i * block_size + j];
        }
        col_count += ncols;
        mat_count += ncols * block_size * block_size;

        for (Arccore::Integer i = 0; i < block_size; ++i) {
          const bool success = targetImpl.setMatrixValues(
              row * block_size + i, ncols * block_size, indices.data(), values[i].data());

          if (not success) {
            throw Arccore::FatalErrorException(A_FUNCINFO,
                Arccore::String::format("Cannot set PETSc Matrix Values for row ", row));
          }
        }
      }
    }
#endif
  }

  if (not targetImpl.assemble()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc assembling failed");
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_PETSc_MatrixConverter);
