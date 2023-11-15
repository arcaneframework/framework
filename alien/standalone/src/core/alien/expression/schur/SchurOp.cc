// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*!
 * \file SchurOp.cc
 * \brief SchurOp.cc
 */


#include <alien/utils/Precomp.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/handlers/scalar/CSRModifierViewT.h>

#include "SchurOp.h"
#include "SchurBlock.h"
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SchurOp::SchurOp(IMatrix& A, IVector& B)
: m_A(A)
, m_B(B)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SchurOp::eErrorType
SchurOp::computePrimarySystem(IMatrix& pA, IVector& pB) const
{
  MatrixImpl& csr_A = m_A.impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  VectorImpl& csr_B = m_B.impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  MatrixImpl& csr_pA = pA.impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  VectorImpl& csr_pB = pB.impl()->get<Alien::BackEnd::tag::simplecsr>(true);

  bool is_fixed_block_size = csr_A.block() != nullptr;
  bool p_is_fixed_block_size = csr_pA.block() != nullptr;
  if (is_fixed_block_size) {
    if (p_is_fixed_block_size)
      return _apply_schur(csr_A.block()->size(),
                          csr_A,
                          csr_B,
                          csr_pA.block()->size(),
                          csr_pA,
                          csr_pB);
    else
      return _apply_schur(csr_A.block()->size(),
                          csr_A,
                          csr_B,
                          csr_pA.vblock(),
                          csr_pA,
                          csr_pB);
  }
  else {
    if (p_is_fixed_block_size)
      return _apply_schur(csr_A.vblock(),
                          csr_A,
                          csr_B,
                          csr_pA.block()->size(),
                          csr_pA,
                          csr_pB);
    else
      return _apply_schur(csr_A.vblock(),
                          csr_A,
                          csr_B,
                          csr_pA.vblock(),
                          csr_pA,
                          csr_pB);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
SchurOp::eErrorType
SchurOp::_apply_schur(Integer block_size,
                      MatrixImpl& A,
                      VectorImpl& B,
                      Integer p_block_size,
                      MatrixImpl& pA,
                      VectorImpl& pB) const
{
  pA.copyProfile(A);
  pA.allocate();

  bool is_parallel = A.isParallel();

  // clang-format off
  CSRConstViewT<MatrixImpl> view(A);
  auto nrows  = view.nrows() ;
  auto kcol   = view.kcol() ;
  auto dcol   = view.dcol() ;
  auto cols   = view.cols() ;
  auto values = view.data() ;

  CSRModifierViewT<MatrixImpl> modifier(pA);
  auto p_nrows  = modifier.nrows() ;
  auto p_kcol   = modifier.kcol() ;
  auto p_dcol   = modifier.dcol() ;
  auto p_cols   = modifier.cols() ;
  auto p_values = modifier.data() ;
  // clang-format on
  if (is_parallel) {
  }
  else {
    for (std::size_t irow = 0; irow < nrows; ++irow) {
      auto diag_offset = dcol[irow];
    }
  }
  return NoError;
}

SchurOp::eErrorType
SchurOp::_apply_schur(Integer block_size,
                      MatrixImpl& A,
                      VectorImpl& B,
                      VBlock const* p_vblock,
                      MatrixImpl& pA,
                      VectorImpl& pB) const
{
  return NoError;
}

SchurOp::eErrorType
SchurOp::_apply_schur(VBlock const* vblock,
                      MatrixImpl& A,
                      VectorImpl& B,
                      Integer p_block_size,
                      MatrixImpl& pA,
                      VectorImpl& pB) const
{
#ifdef ALIEN_USE_EIGEN3
  pA.copyProfile(A);
  pA.allocate();
  auto p_block2d_size = p_block_size * p_block_size;

  bool is_parallel = A.isParallel();
  auto local_offset = A.getLocalOffset();
  if (is_parallel) {
    auto alloc_size = A.getAllocSize();
    B.resize(alloc_size);
  }

  // clang-format off
  CSRModifierViewT<MatrixImpl> view(A);
  auto nrows  = view.nrows() ;
  auto kcol   = view.kcol() ;
  auto dcol   = view.dcol() ;
  auto cols   = view.cols() ;
  auto values = view.data() ;
  auto bcols  = A.getProfile().getBlockCols() ;
  auto brows  = A.getProfile().getBlockRowOffset() ;

  CSRModifierViewT<MatrixImpl> modifier(pA);
  auto p_nrows  = modifier.nrows() ;
  auto p_kcol   = modifier.kcol() ;
  auto p_dcol   = modifier.dcol() ;
  auto p_cols   = modifier.cols() ;
  auto p_values = modifier.data() ;
  // clang-format on

  for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
    auto diag_offset = dcol[irow];
    auto irow_blk_size = vblock->size(local_offset + irow);
    auto irow_offset = B.vblockImpl().offset(local_offset + irow);
    if (irow_blk_size > p_block_size) {
      Array2View<Real> diag(values + bcols[diag_offset], irow_blk_size, irow_blk_size);
      SchurBlock2D diag2d(diag, p_block_size);
      ArrayView<Real> bv(irow_blk_size, B.data() + irow_offset);
      SchurBlock1D b1d(bv, p_block_size);
      SchurAlgo::compute(diag2d, b1d);
    }
  }
  if (is_parallel) {
    auto& dist_struct_info = A.getDistStructInfo();
    auto& local_row_size = dist_struct_info.m_local_row_size;
    auto const& cols = dist_struct_info.m_cols;
    auto const& block_sizes = dist_struct_info.m_block_sizes;
    auto const& block_offsets = dist_struct_info.m_block_offsets;

    SimpleCSRInternal::SendRecvOp<Real> op(B.data(),
                                           dist_struct_info.m_send_info,
                                           A.getSendPolicy(),
                                           B.data(),
                                           dist_struct_info.m_recv_info,
                                           A.getRecvPolicy(),
                                           A.getParallelMng(),
                                           A.traceMng(),
                                           block_sizes,
                                           block_offsets);

    op.start();

    dist_struct_info.computeBlock2DSizesAndOffsets(kcol, dcol, bcols.data());
    auto const& block2d_sizes = dist_struct_info.m_block2d_sizes;
    auto const& block2d_offsets = dist_struct_info.m_block2d_offsets;
    auto ghost_nrow = dist_struct_info.m_ghost_nrow;
    auto ghost_diag_size = dist_struct_info.m_block2d_offsets[nrows + ghost_nrow] - dist_struct_info.m_block2d_offsets[nrows];
    m_ghost_diag_values.resize(ghost_diag_size);
    SimpleCSRInternal::SendRecvOp<Real> diag_op(values,
                                                dist_struct_info.m_send_info,
                                                A.getSendPolicy(),
                                                m_ghost_diag_values.data(),
                                                dist_struct_info.m_recv_info,
                                                A.getRecvPolicy(),
                                                A.getParallelMng(),
                                                A.traceMng(),
                                                block2d_sizes,
                                                block2d_offsets,
                                                true);

    diag_op.start();
    for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
      auto irow_blk_size = block_sizes[irow];
      auto irow_offset = block_offsets[irow];
      ArrayView<Real> bv(irow_blk_size, B.data() + irow_offset);
      SchurBlock1D b1d(bv, p_block_size);
      auto irow_diag_offset = dcol[irow];
      for (int k = kcol[irow]; k < kcol[irow] + local_row_size[irow]; ++k) {

        if (k == irow_diag_offset)
          continue;

        auto col = cols[k];
        auto col_diag_offset = dcol[col];
        auto col_blk_size = block_sizes[col];
        auto col_offset = block_offsets[col];
        if (col_blk_size > p_block_size) {
          Array2View<Real> diag(values + bcols[col_diag_offset], col_blk_size, col_blk_size);
          SchurBlock2D diag2d(diag, p_block_size);

          ArrayView<Real> diag_bv(col_blk_size, (Real*)B.data() + col_offset);
          SchurBlock1D diag_b1d(diag_bv, p_block_size);

          Array2View<Real> off_diag(values + bcols[k], irow_blk_size, col_blk_size);
          SchurBlock2D off_diag2d(off_diag, p_block_size);

          // A11_ki => A11_ki - A12_ki * A22^-1_ii A21_ii
          off_diag2d.block_11() -= off_diag2d.block_12() * diag2d.block_21();

          // b1_k => b1_k - A12_k * A22^-1_i * b2_i
          b1d.block_1() -= off_diag2d.block_12() * diag_b1d.block_2();
        }
      }
    }
    op.end();
    diag_op.end();
    Integer interface_nrow = dist_struct_info.m_interface_nrow;
    ConstArrayView<Integer> row_ids = dist_struct_info.m_interface_rows;
    for (Integer i = 0; i < interface_nrow; ++i) {
      Integer irow = row_ids[i];
      auto irow_blk_size = block_sizes[irow];
      auto irow_offset = block_offsets[irow];
      ArrayView<Real> bv(irow_blk_size, B.data() + irow_offset);
      SchurBlock1D b1d(bv, p_block_size);
      for (int k = kcol[irow] + local_row_size[irow]; k < kcol[irow + 1]; ++k) {

        auto col = cols[k];
        auto col_diag_offset = block2d_offsets[col] - block2d_offsets[nrows];
        auto col_blk_size = block_sizes[col];
        auto col_offset = block_offsets[col];

        if (col_blk_size > p_block_size) {
          Array2View<Real> diag(m_ghost_diag_values.data() + col_diag_offset, col_blk_size, col_blk_size);
          SchurBlock2D diag2d(diag, p_block_size);

          ArrayView<Real> diag_bv(col_blk_size, (Real*)B.data() + col_offset);
          SchurBlock1D diag_b1d(diag_bv, p_block_size);

          Array2View<Real> off_diag(values + bcols[k], irow_blk_size, col_blk_size);
          SchurBlock2D off_diag2d(off_diag, p_block_size);

          // A11_ki => A11_ki - A12_ki * A22^-1_ii A21_ii
          off_diag2d.block_11() -= off_diag2d.block_12() * diag2d.block_21();

          // b1_k => b1_k - A12_k * A22^-1_i * b2_i
          b1d.block_1() -= off_diag2d.block_12() * diag_b1d.block_2();
        }
      }
    }
  }
  else {
    for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
      auto irow_blk_size = vblock->size(local_offset + irow);
      auto irow_offset = B.vblockImpl().offset(local_offset + irow);
      ArrayView<Real> bv(irow_blk_size, B.data() + irow_offset);
      SchurBlock1D b1d(bv, p_block_size);
      auto irow_diag_offset = dcol[irow];
      for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {

        if (k == irow_diag_offset)
          continue;

        auto col = cols[k];
        auto col_diag_offset = dcol[col - local_offset];
        auto col_blk_size = vblock->size(col);
        auto col_offset = B.vblockImpl().offset(col);
        if (col_blk_size > p_block_size) {
          Array2View<Real> diag(values + bcols[col_diag_offset], col_blk_size, col_blk_size);
          SchurBlock2D diag2d(diag, p_block_size);

          ArrayView<Real> diag_bv(col_blk_size, (Real*)B.data() + col_offset);
          SchurBlock1D diag_b1d(diag_bv, p_block_size);

          Array2View<Real> off_diag(values + bcols[k], irow_blk_size, col_blk_size);
          SchurBlock2D off_diag2d(off_diag, p_block_size);

          // A11_ki => A11_ki - A12_ki * A22^-1_ii A21_ii
          off_diag2d.block_11() -= off_diag2d.block_12() * diag2d.block_21();

          // b1_k => b1_k - A12_k * A22^-1_i * b2_i
          b1d.block_1() -= off_diag2d.block_12() * diag_b1d.block_2();
        }
      }
    }

    for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
      auto irow_blk_size = vblock->size(local_offset + irow);
      auto irow_offset = B.vblockImpl().offset(local_offset + irow);
      ArrayView<Real> bv(irow_blk_size, (Real*)B.data() + irow_offset);
      ArrayView<Real> p_bv(p_block_size, pB.data() + irow * p_block_size);
      _copy(bv, p_bv);
      for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {

        auto col = cols[k];
        auto col_blk_size = vblock->size(col);

        Array2View<Real> blk_values((Real*)values + bcols[k], irow_blk_size, col_blk_size);
        Array2View<Real> pblk_values((Real*)p_values + k * p_block2d_size, p_block_size, p_block_size);
        _copy(blk_values, pblk_values);
      }
    }
  }
#endif
  return NoError;
}

SchurOp::eErrorType
SchurOp::_apply_schur(VBlock const* vblock,
                      MatrixImpl& A,
                      VectorImpl& B,
                      VBlock const* p_vblock,
                      MatrixImpl& pA,
                      VectorImpl& pB) const
{
#ifdef ALIEN_USE_EIGEN3
  pA.copyProfile(A);
  pA.allocate();
  bool is_parallel = A.isParallel();
  auto local_offset = A.getLocalOffset();
  if (is_parallel) {
    auto alloc_size = A.getAllocSize();
    B.resize(alloc_size);
  }

  // clang-format off
  CSRModifierViewT<MatrixImpl> view(A);
  auto nrows  = view.nrows() ;
  auto kcol   = view.kcol() ;
  auto dcol   = view.dcol() ;
  auto cols   = view.cols() ;
  auto values = view.data() ;
  auto bcols  = A.getProfile().getBlockCols() ;
  auto brows  = A.getProfile().getBlockRowOffset() ;

  CSRModifierViewT<MatrixImpl> modifier(pA);
  auto p_nrows  = modifier.nrows() ;
  auto p_kcol   = modifier.kcol() ;
  auto p_dcol   = modifier.dcol() ;
  auto p_cols   = modifier.cols() ;
  auto p_values = modifier.data() ;
  auto p_bcols  = pA.getProfile().getBlockCols() ;
  auto p_brows  = pA.getProfile().getBlockRowOffset() ;
  // clang-format on

  for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
    auto diag_offset = dcol[irow];
    auto irow_blk_size = vblock->size(local_offset + irow);
    auto irow_p_blk_size = p_vblock->size(local_offset + irow);
    auto irow_offset = B.vblockImpl().offset(local_offset + irow);
    if (irow_blk_size > irow_p_blk_size) {
      Array2View<Real> diag(values + bcols[diag_offset], irow_blk_size, irow_blk_size);
      SchurBlock2D diag2d(diag, irow_p_blk_size);
      ArrayView<Real> bv(irow_blk_size, B.data() + irow_offset);
      SchurBlock1D b1d(bv, irow_p_blk_size);
      SchurAlgo::compute(diag2d, b1d);
    }
  }

  if (is_parallel) {
    auto& dist_struct_info = A.getDistStructInfo();
    auto& local_row_size = dist_struct_info.m_local_row_size;
    auto const& cols = dist_struct_info.m_cols;
    auto const& block_sizes = dist_struct_info.m_block_sizes;
    auto const& block_offsets = dist_struct_info.m_block_offsets;

    auto& p_dist_struct_info = pA.getDistStructInfo();
    auto const& p_block_sizes = dist_struct_info.m_block_sizes;
    auto const& p_block_offsets = dist_struct_info.m_block_offsets;

    SimpleCSRInternal::SendRecvOp<Real> op(B.data(),
                                           dist_struct_info.m_send_info,
                                           A.getSendPolicy(),
                                           B.data(),
                                           dist_struct_info.m_recv_info,
                                           A.getRecvPolicy(),
                                           A.getParallelMng(),
                                           A.traceMng(),
                                           block_sizes,
                                           block_offsets);

    op.start();

    dist_struct_info.computeBlock2DSizesAndOffsets(kcol, dcol, bcols.data());
    auto const& block2d_sizes = dist_struct_info.m_block2d_sizes;
    auto const& block2d_offsets = dist_struct_info.m_block2d_offsets;
    auto ghost_nrow = dist_struct_info.m_ghost_nrow;
    auto ghost_diag_size = dist_struct_info.m_block2d_offsets[nrows + ghost_nrow] - dist_struct_info.m_block2d_offsets[nrows];
    m_ghost_diag_values.resize(ghost_diag_size);
    SimpleCSRInternal::SendRecvOp<Real> diag_op(values,
                                                dist_struct_info.m_send_info,
                                                A.getSendPolicy(),
                                                m_ghost_diag_values.data(),
                                                dist_struct_info.m_recv_info,
                                                A.getRecvPolicy(),
                                                A.getParallelMng(),
                                                A.traceMng(),
                                                block2d_sizes,
                                                block2d_offsets,
                                                true);

    diag_op.start();

    for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
      auto irow_blk_size = block_sizes[irow];
      auto irow_offset = block_offsets[irow];
      auto irow_p_blk_size = p_vblock->size(local_offset + irow);
      ArrayView<Real> bv(irow_blk_size, B.data() + irow_offset);
      SchurBlock1D b1d(bv, irow_p_blk_size);
      auto irow_diag_offset = dcol[irow];
      for (int k = kcol[irow]; k < kcol[irow] + local_row_size[irow]; ++k) {

        if (k == irow_diag_offset)
          continue;

        auto col = cols[k];
        auto col_diag_offset = dcol[col];
        auto col_blk_size = block_sizes[col];
        auto col_offset = block_offsets[col];
        auto col_p_blk_size = p_vblock->size(local_offset + col);
        if (col_blk_size > col_p_blk_size) {
          Array2View<Real> diag(values + bcols[col_diag_offset], col_blk_size, col_blk_size);
          SchurBlock2D diag2d(diag, col_p_blk_size);

          ArrayView<Real> diag_bv(col_blk_size, (Real*)B.data() + col_offset);
          SchurBlock1D diag_b1d(diag_bv, col_p_blk_size);

          Array2View<Real> off_diag(values + bcols[k], irow_blk_size, col_blk_size);
          SchurBlock2D off_diag2d(off_diag, irow_p_blk_size);

          // A11_ki => A11_ki - A12_ki * A22^-1_ii A21_ii
          off_diag2d.block_11() -= off_diag2d.block_12() * diag2d.block_21();

          // b1_k => b1_k - A12_k * A22^-1_i * b2_i
          b1d.block_1() -= off_diag2d.block_12() * diag_b1d.block_2();
        }
      }
    }

    op.end();
    diag_op.end();

    Integer interface_nrow = dist_struct_info.m_interface_nrow;
    ConstArrayView<Integer> row_ids = dist_struct_info.m_interface_rows;
    for (Integer i = 0; i < interface_nrow; ++i) {
      Integer irow = row_ids[i];
      auto irow_blk_size = block_sizes[irow];
      auto irow_offset = block_offsets[irow];
      auto irow_p_blk_size = p_vblock->size(local_offset + irow);
      ArrayView<Real> bv(irow_blk_size, B.data() + irow_offset);
      SchurBlock1D b1d(bv, irow_p_blk_size);

      for (int k = kcol[irow] + local_row_size[irow]; k < kcol[irow + 1]; ++k) {

        auto col = cols[k];
        auto col_diag_offset = block2d_offsets[col] - block2d_offsets[nrows];
        auto col_blk_size = block_sizes[col];
        auto col_offset = block_offsets[col];
        auto col_p_blk_size = p_block_sizes[col];

        if (col_blk_size > col_p_blk_size) {
          Array2View<Real> diag(m_ghost_diag_values.data() + col_diag_offset, col_blk_size, col_blk_size);
          SchurBlock2D diag2d(diag, col_p_blk_size);

          ArrayView<Real> diag_bv(col_blk_size, (Real*)B.data() + col_offset);
          SchurBlock1D diag_b1d(diag_bv, col_p_blk_size);

          Array2View<Real> off_diag(values + bcols[k], irow_blk_size, col_blk_size);
          SchurBlock2D off_diag2d(off_diag, col_p_blk_size);

          // A11_ki => A11_ki - A12_ki * A22^-1_ii A21_ii
          off_diag2d.block_11() -= off_diag2d.block_12() * diag2d.block_21();

          // b1_k => b1_k - A12_k * A22^-1_i * b2_i
          b1d.block_1() -= off_diag2d.block_12() * diag_b1d.block_2();
        }
      }
    }
    for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
      auto irow_blk_size = vblock->size(local_offset + irow);
      auto irow_offset = B.vblockImpl().offset(local_offset + irow);
      auto irow_p_blk_size = p_vblock->size(local_offset + irow);
      auto irow_p_offset = pB.vblockImpl().offset(local_offset + irow);
      ArrayView<Real> bv(irow_blk_size, (Real*)B.data() + irow_offset);
      ArrayView<Real> p_bv(irow_p_blk_size, pB.data() + irow_p_offset);
      _copy(bv, p_bv);

      for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {

        auto col = cols[k];
        auto col_blk_size = block_sizes[col];
        auto col_p_blk_size = p_block_sizes[col];

        Array2View<Real> blk_values(values + bcols[k], irow_blk_size, col_blk_size);
        Array2View<Real> pblk_values(p_values + p_bcols[k], irow_p_blk_size, col_p_blk_size);
        _copy(blk_values, pblk_values);
      }
    }
  }
  else {
    for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
      auto irow_blk_size = vblock->size(local_offset + irow);
      auto irow_offset = B.vblockImpl().offset(local_offset + irow);
      auto irow_p_blk_size = p_vblock->size(local_offset + irow);
      ArrayView<Real> bv(irow_blk_size, B.data() + irow_offset);
      SchurBlock1D b1d(bv, irow_p_blk_size);
      auto irow_diag_offset = dcol[irow];
      for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {

        if (k == irow_diag_offset)
          continue;

        auto col = cols[k];
        auto col_diag_offset = dcol[col - local_offset];
        auto col_blk_size = vblock->size(col);
        auto col_offset = B.vblockImpl().offset(col);
        auto col_p_blk_size = p_vblock->size(local_offset + col);
        if (col_blk_size > col_p_blk_size) {
          Array2View<Real> diag(values + bcols[col_diag_offset], col_blk_size, col_blk_size);
          SchurBlock2D diag2d(diag, col_p_blk_size);

          ArrayView<Real> diag_bv(col_blk_size, (Real*)B.data() + col_offset);
          SchurBlock1D diag_b1d(diag_bv, col_p_blk_size);

          Array2View<Real> off_diag(values + bcols[k], irow_blk_size, col_blk_size);
          SchurBlock2D off_diag2d(off_diag, col_p_blk_size);

          // A11_ki => A11_ki - A12_ki * A22^-1_ii A21_ii
          off_diag2d.block_11() -= off_diag2d.block_12() * diag2d.block_21();

          // b1_k => b1_k - A12_k * A22^-1_i * b2_i
          b1d.block_1() -= off_diag2d.block_12() * diag_b1d.block_2();
        }
      }
    }
    for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
      auto irow_blk_size = vblock->size(local_offset + irow);
      auto irow_offset = B.vblockImpl().offset(local_offset + irow);
      auto irow_p_blk_size = p_vblock->size(local_offset + irow);
      auto irow_p_offset = pB.vblockImpl().offset(local_offset + irow);
      ArrayView<Real> bv(irow_blk_size, (Real*)B.data() + irow_offset);
      ArrayView<Real> p_bv(irow_p_blk_size, pB.data() + irow_p_offset);
      _copy(bv, p_bv);

      for (int k = kcol[irow]; k < kcol[irow + 1]; ++k) {

        auto col = cols[k];
        auto col_blk_size = vblock->size(col);
        auto col_p_blk_size = p_vblock->size(col);

        Array2View<Real> blk_values((Real*)values + bcols[k], irow_blk_size, col_blk_size);
        Array2View<Real> pblk_values((Real*)p_values + p_bcols[k], irow_p_blk_size, col_p_blk_size);
        _copy(blk_values, pblk_values);
      }
    }
  }
#endif
  return NoError;
}

SchurOp::eErrorType
SchurOp::computeSolutionFromPrimaryUnknowns(IVector const& pX, IVector& X) const
{
  MatrixImpl const& csr_A = m_A.impl()->get<Alien::BackEnd::tag::simplecsr>();

  VectorImpl const& csr_b = m_B.impl()->get<Alien::BackEnd::tag::simplecsr>();
  VectorImpl const& csr_px = pX.impl()->get<Alien::BackEnd::tag::simplecsr>();
  VectorImpl& csr_x = X.impl()->get<Alien::BackEnd::tag::simplecsr>(true);
  if (csr_A.block()) {
    return NoError;
  }
  else {
    if (csr_px.block())
      return _compute_solution(csr_A.vblock(), csr_A, csr_b, csr_px.block()->size(), csr_px, csr_x);
    else
      return _compute_solution(csr_A.vblock(), csr_A, csr_b, csr_px.vblock(), csr_px, csr_x);
  }
}

SchurOp::eErrorType
SchurOp::_compute_solution(VBlock const* vblock,
                           MatrixImpl const& A,
                           VectorImpl const& B,
                           Integer p_block_size,
                           VectorImpl const& pX,
                           VectorImpl& X) const
{

#ifdef ALIEN_USE_EIGEN3
  auto local_offset = A.getLocalOffset();

  // clang-format off
  CSRConstViewT<MatrixImpl> view(A);
  auto nrows  = view.nrows() ;
  auto kcol   = view.kcol() ;
  auto dcol   = view.dcol() ;
  auto cols   = view.cols() ;
  auto values = view.data() ;
  auto bcols  = A.getProfile().getBlockCols() ;
  // clang-format on

  for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
    auto diag_offset = dcol[irow];
    auto irow_blk_size = vblock->size(local_offset + irow);
    auto irow_offset = X.vblockImpl().offset(local_offset + irow);
    Array2View<Real>
    diag((Real*)values + bcols[diag_offset], irow_blk_size, irow_blk_size);
    SchurBlock2D diag2d(diag, p_block_size);
    ConstArrayView<Real> bb(irow_blk_size, B.data() + irow_offset);
    ArrayView<Real> bx(irow_blk_size, X.data() + irow_offset);
    bx.copy(bb);
    SchurBlock1D x1d(bx, p_block_size);
    ArrayView<Real> p_bx(irow_blk_size, (Real*)pX.data() + p_block_size * irow);
    for (int i = 0; i < p_block_size; ++i)
      bx[i] = p_bx[i];
    if (irow_blk_size > p_block_size) {

      // x2 = A22^-1 * b2 - A22^-1 * A21 * x1
      x1d.block_2() = x1d.block_2() - diag2d.block_21() * x1d.block_1();
    }
  }
#endif
  return NoError;
}

SchurOp::eErrorType
SchurOp::_compute_solution(VBlock const* vblock,
                           MatrixImpl const& A,
                           VectorImpl const& B,
                           VBlock const* p_vblock,
                           VectorImpl const& pX,
                           VectorImpl& X) const
{
#ifdef ALIEN_USE_EIGEN3
  auto local_offset = A.getLocalOffset();

  // clang-format off
  CSRConstViewT<MatrixImpl> view(A);
  auto nrows  = view.nrows() ;
  auto kcol   = view.kcol() ;
  auto dcol   = view.dcol() ;
  auto cols   = view.cols() ;
  auto values = view.data() ;
  auto bcols  = A.getProfile().getBlockCols() ;
  // clang-format on

  for (int irow = 0; (std::size_t) irow < nrows; ++irow) {
    auto diag_offset = dcol[irow];
    auto irow_blk_size = vblock->size(local_offset + irow);
    auto irow_offset = X.vblockImpl().offset(local_offset + irow);
    auto irow_p_blk_size = p_vblock->size(local_offset + irow);
    auto irow_p_offset = pX.vblockImpl().offset(local_offset + irow);
    Array2View<Real> diag((Real*)values + bcols[diag_offset], irow_blk_size, irow_blk_size);
    SchurBlock2D diag2d(diag, irow_p_blk_size);
    ConstArrayView<Real> bb(irow_blk_size, B.data() + irow_offset);
    ArrayView<Real> bx(irow_blk_size, X.data() + irow_offset);
    bx.copy(bb);
    SchurBlock1D x1d(bx, irow_p_blk_size);
    ArrayView<Real> p_bx(irow_blk_size, (Real*)pX.data() + irow_p_offset);
    for (int i = 0; i < irow_p_blk_size; ++i)
      bx[i] = p_bx[i];
    if (irow_blk_size > irow_p_blk_size) {

      // x2 = A22^-1 * b2 - A22^-1 * A21 * x1
      x1d.block_2() = x1d.block_2() - diag2d.block_21() * x1d.block_1();
    }
  }
#endif
  return NoError;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
