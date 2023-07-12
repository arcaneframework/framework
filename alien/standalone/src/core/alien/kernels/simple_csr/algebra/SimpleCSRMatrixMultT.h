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

#pragma once

#include <arccore/collections/Array2.h>

#include <alien/handlers/scalar/CSRModifierViewT.h>

/*---------------------------------------------------------------------------*/

namespace Alien::SimpleCSRInternal
{

/*---------------------------------------------------------------------------*/

template <typename ValueT>
SimpleCSRMatrixMultT<ValueT>::SimpleCSRMatrixMultT(const MatrixType& matrix)
: m_matrix_impl(matrix)
{}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::mult(const VectorType& x, VectorType& y) const
{
  if (m_matrix_impl.block()) {
    if (m_matrix_impl.m_is_parallel)
      _parallelMultBlock(x, y);
    else
      _seqMultBlock(x, y);
  }
  else if (m_matrix_impl.vblock()) {
    if (m_matrix_impl.m_is_parallel)
      _parallelMultVariableBlock(x, y);
    else
      _seqMultVariableBlock(x, y);
  }
  else {
    if (m_matrix_impl.m_is_parallel)
      _parallelMult(x, y);
    else
      _seqMult(x, y);
  }
}

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::addLMult(Real alpha, const VectorType& x, VectorType& y) const
{
  _seqAddLMult(alpha, x, y);
}

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::addUMult(Real alpha, const VectorType& x, VectorType& y) const
{
  _seqAddUMult(alpha, x, y);
}

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::mult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const
{
  if (m_matrix_impl.m_is_parallel)
    _parallelMult(x, y);
  else
    _seqMult(x, y);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::_parallelMult(
const VectorType& x_impl, VectorType& y_impl) const
{
  Integer alloc_size = m_matrix_impl.m_local_size + m_matrix_impl.m_ghost_size;
  x_impl.resize(alloc_size);
  Real* y_ptr = y_impl.getDataPtr();
  Real* x_ptr = (Real*)x_impl.getDataPtr();
  ConstArrayView<Real> matrix = m_matrix_impl.m_matrix.getValues();
  // ConstArrayView<Integer> cols2 =
  // m_matrix_impl.m_matrix.getCSRProfile().getCols();
  ConstArrayView<Integer> cols = m_matrix_impl.getDistStructInfo().m_cols;
  ConstArrayView<Integer> row_offset =
  m_matrix_impl.m_matrix.getCSRProfile().getRowOffset();
  SendRecvOp<Real> op(x_ptr, m_matrix_impl.m_matrix_dist_info.m_send_info,
                      m_matrix_impl.m_send_policy, x_ptr, m_matrix_impl.m_matrix_dist_info.m_recv_info,
                      m_matrix_impl.m_recv_policy, m_matrix_impl.m_parallel_mng, m_matrix_impl.m_trace);
  op.start();
  ConstArrayView<Integer> local_row_size =
  m_matrix_impl.m_matrix_dist_info.m_local_row_size;
  for (Integer irow = 0; irow < m_matrix_impl.m_local_size; ++irow) {
    Integer off = row_offset[irow];
    Integer off2 = off + local_row_size[irow];
    Real tmpy = 0.;
    for (Integer j = off; j < off2; ++j) {
      tmpy += matrix[j] * x_ptr[cols[j]];
    }
    y_ptr[irow] = tmpy;
  }
  op.end();

  Integer interface_nrow = m_matrix_impl.m_matrix_dist_info.m_interface_nrow;
  ConstArrayView<Integer> row_ids = m_matrix_impl.m_matrix_dist_info.m_interface_rows;
  for (Integer i = 0; i < interface_nrow; ++i) {
    Integer irow = row_ids[i];
    Integer off = row_offset[irow] + local_row_size[irow];
    Integer off2 = row_offset[irow + 1];
    Real tmpy = 0.;
    for (Integer j = off; j < off2; ++j) {
      tmpy += matrix[j] * x_ptr[cols[j]];
    }
    y_ptr[irow] += tmpy;
  }
}

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::_parallelMult(
const UniqueArray<Real>& x_impl, UniqueArray<Real>& y_impl) const
{
  Real* y_ptr = dataPtr(y_impl);
  Real* x_ptr = (Real*)dataPtr(x_impl);
  ConstArrayView<Real> matrix = m_matrix_impl.m_matrix.getValues();
  ConstArrayView<Integer> cols = m_matrix_impl.getDistStructInfo().m_cols;
  ConstArrayView<Integer> row_offset =
  m_matrix_impl.m_matrix.getCSRProfile().getRowOffset();
  SendRecvOp<Real> op(x_ptr, m_matrix_impl.m_matrix_dist_info.m_send_info,
                      m_matrix_impl.m_send_policy, x_ptr, m_matrix_impl.m_matrix_dist_info.m_recv_info,
                      m_matrix_impl.m_recv_policy, m_matrix_impl.m_parallel_mng, m_matrix_impl.m_trace);
  op.start();
  ConstArrayView<Integer> local_row_size =
  m_matrix_impl.m_matrix_dist_info.m_local_row_size;
  for (Integer irow = 0; irow < m_matrix_impl.m_local_size; ++irow) {
    Integer off = row_offset[irow];
    Integer off2 = off + local_row_size[irow];
    Real tmpy = 0.;
    for (Integer j = off; j < off2; ++j) {
      tmpy += matrix[j] * x_ptr[cols[j]];
    }
    y_ptr[irow] = tmpy;
  }
  op.end();

  Integer interface_nrow = m_matrix_impl.m_matrix_dist_info.m_interface_nrow;
  ConstArrayView<Integer> row_ids = m_matrix_impl.m_matrix_dist_info.m_interface_rows;
  for (Integer i = 0; i < interface_nrow; ++i) {
    Integer irow = row_ids[i];
    Integer off = row_offset[irow] + local_row_size[irow];
    Integer off2 = row_offset[irow + 1];
    Real tmpy = 0.;
    for (Integer j = off; j < off2; ++j) {
      tmpy += matrix[j] * x_ptr[cols[j]];
    }
    y_ptr[irow] += tmpy;
  }
}
/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::_seqMult(const VectorType& x_impl, VectorType& y_impl) const
{
#ifdef ALIEN_USE_PERF_TIMER
  typename MatrixType::SentryType sentry(m_matrix_impl.timer(), "CSR-SPMV");
#endif
  Real* y_ptr = y_impl.getDataPtr();
  Real* x_ptr = (Real*)x_impl.getDataPtr();
  ConstArrayView<Real> matrix = m_matrix_impl.m_matrix.getValues();
  ConstArrayView<Integer> cols = m_matrix_impl.m_matrix.getCSRProfile().getCols();
  ConstArrayView<Integer> row_offset =
  m_matrix_impl.m_matrix.getCSRProfile().getRowOffset();
  for (Integer irow = 0; irow < m_matrix_impl.m_local_size; ++irow) {
    Real tmpy = 0.;
    for (Integer j = row_offset[irow]; j < row_offset[irow + 1]; ++j) {
      tmpy += matrix[j] * x_ptr[cols[j]];
    }
    y_ptr[irow] = tmpy;
  }
}

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::_seqAddLMult(Real alpha, const VectorType& x_impl, VectorType& y_impl) const
{
#ifdef ALIEN_USE_PERF_TIMER
  typename MatrixType::SentryType sentry(m_matrix_impl.timer(), "CSR-AddLMult");
#endif
  Real* y_ptr = y_impl.getDataPtr();
  Real* x_ptr = (Real*)x_impl.getDataPtr();
  ConstArrayView<Real> matrix = m_matrix_impl.m_matrix.getValues();
  ConstArrayView<Integer> cols = m_matrix_impl.m_matrix.getCSRProfile().getCols();
  ConstArrayView<Integer> row_offset = m_matrix_impl.m_matrix.getCSRProfile().getRowOffset();
  auto diag_offset = m_matrix_impl.m_matrix.getCSRProfile().getUpperDiagOffset();
  for (Integer irow = 0; irow < m_matrix_impl.m_local_size; ++irow) {
    Real tmpy = y_ptr[irow];
    for (Integer j = row_offset[irow]; j < diag_offset[irow]; ++j) {
      tmpy += alpha * matrix[j] * x_ptr[cols[j]];
    }
    y_ptr[irow] = tmpy;
  }
}

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::_seqAddUMult(Real alpha, const VectorType& x_impl, VectorType& y_impl) const
{
#ifdef ALIEN_USE_PERF_TIMER
  typename MatrixType::SentryType sentry(m_matrix_impl.timer(), "CSR-AddUMult");
#endif
  Real* y_ptr = y_impl.getDataPtr();
  Real* x_ptr = (Real*)x_impl.getDataPtr();
  ConstArrayView<Real> matrix = m_matrix_impl.m_matrix.getValues();
  ConstArrayView<Integer> cols = m_matrix_impl.m_matrix.getCSRProfile().getCols();
  ConstArrayView<Integer> row_offset = m_matrix_impl.m_matrix.getCSRProfile().getRowOffset();
  auto diag_offset = m_matrix_impl.m_matrix.getCSRProfile().getUpperDiagOffset();
  for (Integer irow = 0; irow < m_matrix_impl.m_local_size; ++irow) {
    Real tmpy = y_ptr[irow];
    for (Integer j = diag_offset[irow] + 1; j < row_offset[irow + 1]; ++j) {
      tmpy += alpha * matrix[j] * x_ptr[cols[j]];
    }
    y_ptr[irow] = tmpy;
  }
}

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::_seqMult(
const UniqueArray<Real>& x_impl, UniqueArray<Real>& y_impl) const
{
  Real* y_ptr = dataPtr(y_impl);
  Real* x_ptr = (Real*)dataPtr(x_impl);
  ConstArrayView<Real> matrix = m_matrix_impl.m_matrix.getValues();
  ConstArrayView<Integer> cols = m_matrix_impl.m_matrix.getCSRProfile().getCols();
  ConstArrayView<Integer> row_offset =
  m_matrix_impl.m_matrix.getCSRProfile().getRowOffset();
  for (Integer irow = 0; irow < m_matrix_impl.m_local_size; ++irow) {
    Real tmpy = 0.;
    for (Integer j = row_offset[irow]; j < row_offset[irow + 1]; ++j) {
      tmpy += matrix[j] * x_ptr[cols[j]];
    }
    y_ptr[irow] = tmpy;
  }
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::_parallelMultBlock(const VectorType& x, VectorType& y) const
{
  Integer alloc_size = m_matrix_impl.m_local_size + m_matrix_impl.m_ghost_size;
  const Integer block_size = m_matrix_impl.block()->size();
  x.resize(alloc_size * block_size);
  ArrayView<Real> _y = y.fullValues();
  ConstArrayView<Real> x_ptr = x.fullValues();
  Real const* matrix = m_matrix_impl.m_matrix.getDataPtr();
  ConstArrayView<Integer> cols = m_matrix_impl.getDistStructInfo().m_cols;
  // ConstArrayView<Integer> cols2 =
  // m_matrix_impl.m_matrix.getCSRProfile().getCols();
  ConstArrayView<Integer> row_offset =
  m_matrix_impl.m_matrix.getCSRProfile().getRowOffset();
  SimpleCSRInternal::SendRecvOp<Real> op(x.getDataPtr(),
                                         m_matrix_impl.m_matrix_dist_info.m_send_info, m_matrix_impl.m_send_policy,
                                         (Real*)x.getDataPtr(), m_matrix_impl.m_matrix_dist_info.m_recv_info,
                                         m_matrix_impl.m_recv_policy, m_matrix_impl.m_parallel_mng, m_matrix_impl.m_trace,
                                         block_size);
  op.start();
  ConstArrayView<Integer> local_row_size =
  m_matrix_impl.m_matrix_dist_info.m_local_row_size;
  for (Integer irow = 0; irow < m_matrix_impl.m_local_size; ++irow) {
    ArrayView<Real> y_ptr = _y.subView(irow * block_size, block_size);
    Integer off = row_offset[irow];
    Integer off2 = off + local_row_size[irow];
    Real const* m = matrix + off * block_size * block_size;
    for (Integer ieq = 0; ieq < block_size; ++ieq)
      y_ptr[ieq] = 0.;
    for (Integer jcol = off; jcol < off2; ++jcol) {
      ConstArrayView<Real> ptr = x_ptr.subView(cols[jcol] * block_size, block_size);
      for (Integer ieq = 0; ieq < block_size; ++ieq)
        for (Integer iu = 0; iu < block_size; ++iu)
          y_ptr[ieq] += m[ieq + block_size * iu] * ptr[iu];
      m += block_size * block_size;
    }
    // y_ptr += block_size;
  }
  op.end();

  Integer interface_nrow = m_matrix_impl.m_matrix_dist_info.m_interface_nrow;
  ConstArrayView<Integer> row_ids = m_matrix_impl.m_matrix_dist_info.m_interface_rows;
  ArrayView<Real> y_ptr = _y;
  for (Integer i = 0; i < interface_nrow; ++i) {
    Integer irow = row_ids[i];
    ArrayView<Real> yptr = y_ptr.subView(irow * block_size, block_size);
    Integer off = row_offset[irow] + local_row_size[irow];
    Integer off2 = row_offset[irow + 1];
    Real const* m = matrix + off * block_size * block_size;
    for (Integer jcol = off; jcol < off2; ++jcol) {
      ConstArrayView<Real> ptr = x_ptr.subView(cols[jcol] * block_size, block_size);
      for (Integer ieq = 0; ieq < block_size; ++ieq)
        for (Integer iu = 0; iu < block_size; ++iu)
          yptr[ieq] += m[ieq + block_size * iu] * ptr[iu];
      m += block_size * block_size;
    }
  }
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::_seqMultBlock(const VectorType& x, VectorType& y) const
{
  Real* y_ptr = y.getDataPtr();
  Real const* x_ptr = x.getDataPtr();
  Real const* matrix = m_matrix_impl.m_matrix.getDataPtr();
  const Integer block_size = m_matrix_impl.block()->size();
  ConstArrayView<Integer> cols = m_matrix_impl.m_matrix.getCSRProfile().getCols();
  ConstArrayView<Integer> row_offset =
  m_matrix_impl.m_matrix.getCSRProfile().getRowOffset();
  for (Integer irow = 0; irow < m_matrix_impl.m_local_size; ++irow) {
    Integer off = row_offset[irow];
    Integer off2 = row_offset[irow + 1];
    Real const* m = matrix + off * block_size * block_size;
    for (Integer ieq = 0; ieq < block_size; ++ieq)
      y_ptr[ieq] = 0.;
    for (Integer jcol = off; jcol < off2; ++jcol) {
      Real const* ptr = x_ptr + cols[jcol] * block_size;
      for (Integer ieq = 0; ieq < block_size; ++ieq)
        for (Integer iu = 0; iu < block_size; ++iu)
          y_ptr[ieq] += m[ieq + block_size * iu] * ptr[iu];
      m += block_size * block_size;
    }
    y_ptr += block_size;
  }
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::_parallelMultVariableBlock(
const VectorType& x_impl, VectorType& y_impl) const
{
  // alien_info([&] { cout()<<"_parallelMultVariableBlock";}) ;

  ArrayView<ValueT> y = y_impl.fullValues();

  ConstArrayView<Integer> block_sizes = m_matrix_impl.getDistStructInfo().m_block_sizes;
  ConstArrayView<Integer> block_offsets =
  m_matrix_impl.getDistStructInfo().m_block_offsets;
  {
    const Integer last = block_offsets.size() - 1;
    x_impl.resize(block_offsets[last] + block_sizes[last]);
  }

  const ValueT* x_ptr = x_impl.getDataPtr();
  const ValueT* matrix_ptr = m_matrix_impl.m_matrix.getDataPtr();

  ConstArrayView<Integer> row_offset =
  m_matrix_impl.m_matrix.getCSRProfile().getRowOffset();
  ConstArrayView<Integer> cols = m_matrix_impl.getDistStructInfo().m_cols;
  ConstArrayView<Integer> block_cols =
  m_matrix_impl.m_matrix.getCSRProfile().getBlockCols();

  SendRecvOp<Real> op(x_ptr, m_matrix_impl.m_matrix_dist_info.m_send_info,
                      m_matrix_impl.m_send_policy, (ValueT*)x_ptr,
                      m_matrix_impl.m_matrix_dist_info.m_recv_info, m_matrix_impl.m_recv_policy,
                      m_matrix_impl.m_parallel_mng, m_matrix_impl.m_trace, block_sizes, block_offsets);

  op.start();

  UniqueArray<ValueT> tmpy;
  tmpy.reserve(m_matrix_impl.vblock()->maxBlockSize());

  ConstArrayView<Integer> local_row_size =
  m_matrix_impl.m_matrix_dist_info.m_local_row_size;
  for (Integer irow = 0; irow < m_matrix_impl.m_local_size; ++irow) {
    Integer off = row_offset[irow];
    Integer off2 = off + local_row_size[irow];
    const Integer block_size_row = block_sizes[irow];
    tmpy.resize(block_size_row);
    tmpy.fill(ValueT());
    for (Integer j = off; j < off2; ++j) {
      const Integer col = cols[j];
      const Integer block_size_col = block_sizes[col];
      ConstArrayView<ValueT> x(block_size_col, x_ptr + block_offsets[col]);
      ConstArray2View<ValueT> matrix(
      matrix_ptr + block_cols[j], block_size_row, block_size_col);
      for (Integer krow = 0; krow < block_size_row; ++krow) {
        for (Integer kcol = 0; kcol < block_size_col; ++kcol) {
          tmpy[krow] += matrix[krow][kcol] * x[kcol];
        }
      }
    }
    y.subView(block_offsets[irow], block_size_row).copy(tmpy);
  }

  op.end();

  Integer interface_nrow = m_matrix_impl.m_matrix_dist_info.m_interface_nrow;
  ConstArrayView<Integer> row_ids = m_matrix_impl.m_matrix_dist_info.m_interface_rows;
  for (Integer i = 0; i < interface_nrow; ++i) {
    Integer irow = row_ids[i];
    Integer off = row_offset[irow] + local_row_size[irow];
    Integer off2 = row_offset[irow + 1];
    const Integer block_size_row = block_sizes[irow];
    tmpy.resize(block_size_row);
    tmpy.fill(ValueT());
    for (Integer j = off; j < off2; ++j) {
      const Integer col = cols[j];
      const Integer block_size_col = block_sizes[col];
      ConstArrayView<ValueT> x(block_size_col, x_ptr + block_offsets[col]);
      ConstArray2View<ValueT> matrix(
      matrix_ptr + block_cols[j], block_size_row, block_size_col);
      for (Integer krow = 0; krow < block_size_row; ++krow) {
        for (Integer kcol = 0; kcol < block_size_col; ++kcol) {
          tmpy[krow] += matrix[krow][kcol] * x[kcol];
        }
      }
    }
    ArrayView<ValueT> y_view = y.subView(block_offsets[irow], block_size_row);
    for (Integer k = 0; k < block_size_row; ++k)
      y_view[k] += tmpy[k];
  }
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::_seqMultVariableBlock(
const VectorType& x_impl, VectorType& y_impl) const
{
  // alien_info([&] { cout()<<"_seqMultVariableBlock";}) ;

  ArrayView<ValueT> y = y_impl.fullValues();

  const ValueT* x_ptr = x_impl.getDataPtr();
  const ValueT* matrix_ptr = m_matrix_impl.m_matrix.getDataPtr();

  ConstArrayView<Integer> row_offset =
  m_matrix_impl.m_matrix.getCSRProfile().getRowOffset();
  ConstArrayView<Integer> cols = m_matrix_impl.m_matrix.getCSRProfile().getCols();
  ConstArrayView<Integer> block_cols =
  m_matrix_impl.m_matrix.getCSRProfile().getBlockCols();
  const VBlock* block = m_matrix_impl.vblock();
  const VBlockImpl& block_infos = x_impl.vblockImpl();

  UniqueArray<ValueT> tmpy;
  tmpy.reserve(block->maxBlockSize());
  for (Integer irow = 0; irow < m_matrix_impl.m_local_size;
       ++irow) // Attention, c'est local !!!!
  {
    const Integer block_size_row = block->size(irow);
    tmpy.resize(block_size_row);
    tmpy.fill(ValueT());
    for (Integer j = row_offset[irow]; j < row_offset[irow + 1]; ++j) {
      const Integer col = cols[j];
      const Integer block_size_col = block->size(col);
      ConstArrayView<ValueT> x(block_size_col, x_ptr + block_infos.offset(col));
      ConstArray2View<ValueT> matrix(
      matrix_ptr + block_cols[j], block_size_row, block_size_col);
      for (Integer krow = 0; krow < block_size_row; ++krow) {
        for (Integer kcol = 0; kcol < block_size_col; ++kcol) {
          tmpy[krow] += matrix[krow][kcol] * x[kcol];
        }
      }
    }
    y.subView(block_infos.offset(irow), block_size_row).copy(tmpy);
  }
}

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::multInvDiag(VectorType& y) const
{
  Real* y_ptr = y.getDataPtr();
  CSRConstViewT<MatrixType> view(m_matrix_impl);
  // clang-format off
  auto nrows  = view.nrows() ;
  auto kcol   = view.kcol() ;
  auto dcol   = view.dcol() ;
  auto cols   = view.cols() ;
  auto values = view.data() ;
  // clang-format on
  for (Integer irow = 0; irow < nrows; ++irow)
    y_ptr[irow] = y_ptr[irow] / values[dcol[irow]];
}

template <typename ValueT>
void SimpleCSRMatrixMultT<ValueT>::computeInvDiag(VectorType& y) const
{
  Real* y_ptr = y.getDataPtr();

  CSRConstViewT<MatrixType> view(m_matrix_impl);
  // clang-format off
  auto nrows  = view.nrows() ;
  auto kcol   = view.kcol() ;
  auto dcol   = view.dcol() ;
  auto cols   = view.cols() ;
  auto values = view.data() ;
  // clang-format on
  for (Integer irow = 0; irow < nrows; ++irow)
    y_ptr[irow] = 1. / values[dcol[irow]];
}

/*---------------------------------------------------------------------------*/

} // namespace Alien::SimpleCSRInternal

/*---------------------------------------------------------------------------*/
