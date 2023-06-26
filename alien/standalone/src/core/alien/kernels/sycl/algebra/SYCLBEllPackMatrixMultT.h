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

#include <alien/kernels/sycl/data/SendRecvOp.h>
/*---------------------------------------------------------------------------*/

namespace Alien::SYCLInternal
{

/*---------------------------------------------------------------------------*/

template <typename ValueT>
SYCLBEllPackMatrixMultT<ValueT>::SYCLBEllPackMatrixMultT(const MatrixType& matrix)
: m_matrix_impl(matrix)
{}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SYCLBEllPackMatrixMultT<ValueT>::mult(const VectorType& x, VectorType& y) const
{
  if (m_matrix_impl.m_is_parallel)
    _parallelMult(x, y);
  else
    _seqMult(x, y);
}

template <typename ValueT>
void SYCLBEllPackMatrixMultT<ValueT>::addLMult(Real alpha, const VectorType& x, VectorType& y) const
{
#ifdef ALIEN_USE_PERF_TIMER
  typename MatrixType::SentryType sentry(m_matrix_impl.timer(), "SYCL-AddLMult");
#endif
  m_matrix_impl.addLMult(alpha, x, y);
}

template <typename ValueT>
void SYCLBEllPackMatrixMultT<ValueT>::addUMult(Real alpha, const VectorType& x, VectorType& y) const
{
#ifdef ALIEN_USE_PERF_TIMER
  typename MatrixType::SentryType sentry(m_matrix_impl.timer(), "SYCL-AddUMult");
#endif
  m_matrix_impl.addUMult(alpha, x, y);
}

template <typename ValueT>
void SYCLBEllPackMatrixMultT<ValueT>::mult(const UniqueArray<Real>& x, UniqueArray<Real>& y) const
{
#ifdef ALIEN_USE_PERF_TIMER
  typename MatrixType::SentryType sentry(m_matrix_impl.timer(), "SYCL-SPMV");
#endif
  if (m_matrix_impl.m_is_parallel)
    _parallelMult(x, y);
  else
    _seqMult(x, y);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SYCLBEllPackMatrixMultT<ValueT>::_parallelMult(
const VectorType& x_impl, VectorType& y_impl) const
{
  //Alien::alien_debug([&] {Alien::cout() << "SYCL PARALLEL MULT : "<<m_matrix_impl.getGhostSize();});
  //Universe().traceMng()->flush() ;
  SYCLSendRecvOp<ValueT> op(x_impl.internal()->values(),
                            m_matrix_impl.m_matrix_dist_info.m_send_info,
                            m_matrix_impl.internal()->getSendIds(),
                            m_matrix_impl.m_send_policy,
                            x_impl.internal()->ghostValues(m_matrix_impl.getGhostSize()),
                            m_matrix_impl.m_matrix_dist_info.m_recv_info,
                            m_matrix_impl.internal()->getRecvIds(),
                            m_matrix_impl.m_recv_policy,
                            m_matrix_impl.m_parallel_mng,
                            m_matrix_impl.m_trace);

  op.start();

  m_matrix_impl.mult(x_impl, y_impl);

  op.end();

  m_matrix_impl.endDistMult(x_impl, y_impl);

  //Alien::alien_debug([&] {Alien::cout() << "End SYCL PARALLEL MULT";});
  //Universe().traceMng()->flush() ;
}

template <typename ValueT>
void SYCLBEllPackMatrixMultT<ValueT>::_parallelMult(
const UniqueArray<Real>& x_impl, UniqueArray<Real>& y_impl) const
{
#ifdef ENABLE_MPI_SYCL
  Real* y_ptr = dataPtr(y_impl);
  Real* x_ptr = (Real*)dataPtr(x_impl);
  ConstArrayView<Real> matrix = m_matrix_impl.m_matrix.getValues();
  ConstArrayView<Integer> cols = m_matrix_impl.getDistStructInfo().m_cols;
  ConstArrayView<Integer> row_offset =
  m_matrix_impl.m_matrix.getProfile().getRowOffset();
  SendRecvOp<Real> op(x_ptr, m_matrix_impl.m_matrix_dist_info.m_send_info,
                      m_matrix_impl.m_send_policy, x_ptr, m_matrix_impl.m_matrix_dist_info.m_recv_info,
                      m_matrix_impl.m_recv_policy, m_matrix_impl.m_parallel_mng, m_matrix_impl.m_trace);
  op.start();
#endif

  //m_matrix_impl.mult(x_impl,y_impl) ;

#ifdef ENABLE_MPI_SYCL
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
#endif
}
/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SYCLBEllPackMatrixMultT<ValueT>::_seqMult(const VectorType& x_impl, VectorType& y_impl) const
{
  m_matrix_impl.mult(x_impl, y_impl);
}

template <typename ValueT>
void SYCLBEllPackMatrixMultT<ValueT>::_seqMult(
const UniqueArray<Real>& x_impl, UniqueArray<Real>& y_impl) const
{
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void SYCLBEllPackMatrixMultT<ValueT>::multInvDiag(VectorType& y) const
{
  m_matrix_impl.multInvDiag(y);
}

template <typename ValueT>
void SYCLBEllPackMatrixMultT<ValueT>::computeInvDiag(VectorType& y) const
{
  m_matrix_impl.computeInvDiag(y);
}

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

} // namespace Alien::SYCLInternal

/*---------------------------------------------------------------------------*/
