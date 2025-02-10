// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LUSendRecvOp.h                                              (C) 2000-2025 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#pragma once

#include <vector>
#include <map>

#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/utils/Precomp.h>
#include <alien/utils/Trace.h>

#include <arccore/message_passing/Messages.h>
#include <arccore/message_passing/Request.h>

#include <alien/handlers/scalar/CSRModifierViewT.h>
#include <alien/kernels/simple_csr/SendRecvOp.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>

namespace Alien
{
class MatrixDistribution;
}

namespace Alien::SYCLInternal
{

#ifndef USE_SYCL2020
  using namespace cl ;
#endif

template <typename MatrixT>
class SYCLLUSendRecvOp
{
 public:
  // clang-format off
  typedef MatrixT                               MatrixType;
  typedef typename MatrixType::ValueType        ValueType;

  typedef sycl::buffer<ValueType, 1>        ValueBufferType ;

  typedef sycl::buffer<int>                 IndexBufferType ;
  typedef std::unique_ptr<IndexBufferType>      IndexBufferPtrType ;
  // clang-format on


  SYCLLUSendRecvOp(MatrixType& matrix,
               MatrixDistribution& distribution,
               std::vector<int>& work,
               Arccore::ITraceMng* trace_mng = nullptr)
  : m_matrix(matrix)
  , m_distribution(distribution)
  , m_work(work)
  , m_send_info(matrix.getDistStructInfo().m_send_info)
  , m_recv_info(matrix.getDistStructInfo().m_recv_info)
  , m_parallel_mng(matrix.getParallelMng())
  , m_trace(trace_mng)
  {
    initSendRecvConnectivity();
  }

  virtual ~SYCLLUSendRecvOp()
  {
#ifdef USE_SYCL_USM
    auto& queue = SYCLEnv::instance()->internal()->queue();
    for(auto& buf : m_recv_lu_buffer)
       sycl::free(buf, queue);
    for(auto& buf : m_recv_lu_ibuffer)
       sycl::free(buf, queue);
    for(auto& buf : m_send_lu_buffer)
       sycl::free(buf, queue);
    for(auto& buf : m_send_lu_ibuffer)
       sycl::free(buf, queue);
#endif
  }


  void initSendRecvConnectivity()
  {

    CSRConstViewT<MatrixT> view(m_matrix);
    // clang-format off
    auto nrows  = view.nrows() ;
    auto kcol   = view.kcol() ;
    //auto dcol   = view.dcol() ;
    auto cols   = view.cols() ;
    // clang-format on
    auto& local_row_size = m_matrix.getDistStructInfo().m_local_row_size;

    //int my_rank = m_parallel_mng->commRank();

    m_mpi_ext_inv_ids.resize(m_recv_info.m_first_upper_neighb);
    for (int ineighb = 0; ineighb < m_recv_info.m_first_upper_neighb; ++ineighb) {
      std::map<int, int>& inv_ids = m_mpi_ext_inv_ids[ineighb];
      for (int i = m_recv_info.m_ids_offset[ineighb]; i < m_recv_info.m_ids_offset[ineighb]; ++i) {
        inv_ids[m_recv_info.m_uids[i]] = i;
      }
    }
    std::size_t recv_uids_size = m_recv_info.m_uids.size();
    std::vector<int> conn_size(recv_uids_size);
    std::fill(conn_size.begin(), conn_size.end(), 0);
    for (int irow = 0; irow < nrows; ++irow) {
      for (int k = kcol[irow] + local_row_size[irow]; k < kcol[irow + 1]; ++k) {
        ++conn_size[cols[k] - nrows];
      }
    }
    m_recv_connectivity_ids_ptr.resize(recv_uids_size + 1);
    m_recv_connectivity_ids_ptr[0] = 0;
    for (std::size_t i = 0; i < recv_uids_size; ++i)
      m_recv_connectivity_ids_ptr[i + 1] = m_recv_connectivity_ids_ptr[i] + conn_size[i];
    std::size_t total_conn_size = m_recv_connectivity_ids_ptr[recv_uids_size];
    m_recv_connectivity_ids.resize(total_conn_size);
    m_recv_connectivity_krow.resize(total_conn_size);
    std::fill(conn_size.begin(), conn_size.end(), 0);
    for (int irow = 0; irow < nrows; ++irow) {
      for (int k = kcol[irow] + local_row_size[irow]; k < kcol[irow + 1]; ++k) {
        int col = cols[k];
        int id = col - nrows;
        m_recv_connectivity_ids[m_recv_connectivity_ids_ptr[id] + conn_size[id]] = irow;
        m_recv_connectivity_krow[m_recv_connectivity_ids_ptr[id] + conn_size[id]] = k;
        ++conn_size[id];
      }
    }
  }

  void sendUpperNeighbLUData(ValueType* values)
  {
    CSRModifierViewT<MatrixType> modifier(m_matrix);
    // clang-format off
    auto nrows  = modifier.nrows() ;
    //auto nnz    = modifier.nnz() ;
    auto kcol   = modifier.kcol() ;
    auto dcol   = modifier.dcol() ;
    auto cols   = modifier.cols() ;
    //auto values = modifier.data() ;
    // clang-format on

    auto max_row_size = m_matrix.getProfile().getMaxRowSize();
    auto& local_row_size = m_matrix.getDistStructInfo().m_local_row_size;

    m_send_lu_ibuffer.resize(m_send_info.m_num_neighbours - m_send_info.m_first_upper_neighb);
    m_send_lu_buffer.resize(m_send_info.m_num_neighbours - m_send_info.m_first_upper_neighb);
    for (int ineighb = m_send_info.m_first_upper_neighb; ineighb < m_send_info.m_num_neighbours; ++ineighb) {
      int neighb = m_send_info.m_ranks[ineighb];
      auto& ibuffer = m_send_lu_ibuffer[ineighb - m_send_info.m_first_upper_neighb];
      auto& buffer = m_send_lu_buffer[ineighb - m_send_info.m_first_upper_neighb];
      int nb_send_rows = m_send_info.m_ids_offset[ineighb + 1] - m_send_info.m_ids_offset[ineighb];
#ifdef USE_SYCL_USM
      int icount = 0 ;
      int count = 0 ;
      ibuffer = sycl::malloc_shared<int>(nb_send_rows * max_row_size, queue);
      buffer = sycl::malloc_shared<ValueType>(nb_send_rows * max_row_size, queue);
#else
      buffer.clear();
      buffer.reserve(nb_send_rows * max_row_size);
      ibuffer.clear();
      ibuffer.reserve(nb_send_rows * max_row_size);
#endif
      for (int i = m_send_info.m_ids_offset[ineighb]; i < m_send_info.m_ids_offset[ineighb + 1]; ++i) {
        int irow = m_send_info.m_ids[i];
        int lrow_size = local_row_size[irow];
        int int_row_size = kcol[irow] + lrow_size - dcol[irow];
        int ext_row_size = kcol[irow + 1] - kcol[irow] - lrow_size;
#ifdef USE_SYCL_USM
        ibuffer[icount++] = int_row_size;
        ibuffer[icount++] = ext_row_size;
#else
        ibuffer.push_back(int_row_size);
        ibuffer.push_back(ext_row_size);
#endif
        for (int k = dcol[irow]; k < kcol[irow] + lrow_size; ++k) {
#ifdef USE_SYCL_USM
          buffer[count++] = values[k];
          ibuffer[icount++] = cols[k]);
#else
          buffer.push_back(values[k]);
          ibuffer.push_back(cols[k]);
#endif
        }
        for (int k = kcol[irow] + lrow_size; k < kcol[irow + 1]; ++k) {
#ifdef USE_SYCL_USM
          buffer[count++] = values[k];
          ibuffer[icount++] = m_recv_info.m_uids[cols[k] - nrows];
#else
          buffer.push_back(values[k]);
          ibuffer.push_back(m_recv_info.m_uids[cols[k] - nrows]);
#endif
        }
      }
      UniqueArray<int> counts(2);
#ifdef USE_SYCL_USM
      counts[0] = icount;
      counts[1] = count;
      Arccore::MessagePassing::mpSend(m_parallel_mng, counts, neighb);
      Arccore::MessagePassing::mpSend(m_parallel_mng, ArrayView<int>(counts[0],ibuffer), neighb);
      Arccore::MessagePassing::mpSend(m_parallel_mng, ArrayView<ValueType>(counts[1],buffer), neighb);
#else
      counts[0] = (int) ibuffer.size();
      counts[1] = (int) buffer.size();
      Arccore::MessagePassing::mpSend(m_parallel_mng, counts, neighb);
      Arccore::MessagePassing::mpSend(m_parallel_mng, ArrayView<int>(counts[0],ibuffer.data()), neighb);
      Arccore::MessagePassing::mpSend(m_parallel_mng, ArrayView<ValueType>(counts[1],buffer.data()), neighb);
#endif
    }
  }

  void recvLowerNeighbLUData(ValueType* values)
  {
    CSRModifierViewT<MatrixT> modifier(m_matrix);
    // clang-format off
    auto nrows  = modifier.nrows() ;
    auto nnz    = modifier.nnz() ;
    auto kcol   = modifier.kcol() ;
    auto dcol   = modifier.dcol() ;
    auto cols   = modifier.cols() ;
    //auto values = modifier.data() ;
    // clang-format on
    auto& local_row_size = m_matrix.getDistStructInfo().m_local_row_size;
    auto const& distribution = m_distribution.rowDistribution();

    int my_rank = m_parallel_mng->commRank();
    int my_domain_offset = distribution.offset(my_rank);

    m_recv_lu_ibuffer.resize(m_recv_info.m_first_upper_neighb);
    m_recv_lu_buffer.resize(m_recv_info.m_first_upper_neighb);
    for (int ineighb = 0; ineighb < m_recv_info.m_first_upper_neighb; ++ineighb) {
      int neighb = m_recv_info.m_ranks[ineighb];
      UniqueArray<int> counts(2);
      Arccore::MessagePassing::mpReceive(m_parallel_mng, counts, neighb);
      auto& ibuffer = m_recv_lu_ibuffer[ineighb];
      auto& buffer = m_recv_lu_buffer[ineighb];
#ifdef USE_SYCL_USM
      ibuffer = sycl::malloc_shared<int>(counts[0], queue);
      buffer = sycl::malloc_shared<int>(counts[1], queue);
      Arccore::MessagePassing::mpReceive(m_parallel_mng, ArrayView<int>(counts[0],ibuffer), neighb);
      Arccore::MessagePassing::mpReceive(m_parallel_mng, ArrayView<ValueType>(counts[1],buffer), neighb);
#else
      ibuffer.resize(counts[0]);
      buffer.resize(counts[1]);
      Arccore::MessagePassing::mpReceive(m_parallel_mng, ArrayView<int>(counts[0],ibuffer.data()), neighb);
      Arccore::MessagePassing::mpReceive(m_parallel_mng, ArrayView<ValueType>(counts[1],buffer.data()), neighb);
#endif
      int icount = 0;
      int icount2 = 0;
      for (int i = m_recv_info.m_ids_offset[ineighb]; i < m_recv_info.m_ids_offset[ineighb + 1]; ++i) {
        int irow = i - nrows;
        int int_row_size = ibuffer[icount++];
        int ext_row_size = ibuffer[icount++];
        for (int conn_k = m_recv_connectivity_ids_ptr[irow]; conn_k < m_recv_connectivity_ids_ptr[irow + 1]; ++conn_k) {
          int conn_row = m_recv_connectivity_ids[conn_k];
          int krow = m_recv_connectivity_krow[conn_k];
          for (int k = krow + 1; k < kcol[conn_row + 1]; ++k) {
            m_work[cols[k]] = k;
          }
          for (int k = kcol[conn_row]; k < kcol[conn_row] + local_row_size[conn_row]; ++k) {
            m_work[cols[k]] = k;
          }

          std::map<int, int>& inv_ids = m_mpi_ext_inv_ids[ineighb];
          ValueType aik = values[krow] / buffer[icount2]; // aik = aik/akk
          //MatrixDataType aik = mpi_ext_values[krow] / buffer[icount2 ]; // aik = aik/akk
          values[krow] = aik;
          for (int k = 1; k < int_row_size; ++k) {
            int uid = ibuffer[icount + k];
            std::map<int, int>::iterator iter = inv_ids.find(uid);
            if (iter != inv_ids.end()) {
              int lid = iter->second;
              int kj = m_work[lid];
              if (kj != -1) {
                values[kj] -= aik * buffer[icount2 + k]; // aij = aij - aik*akj
              }
            }
          }
          for (int k = 0; k < ext_row_size; ++k) {
            int uid = ibuffer[icount + int_row_size + k];
            int owner = distribution.owner(uid);
            if (owner == my_rank) {
              int lid = uid - my_domain_offset;
              int kj = m_work[lid];
              if (kj != -1) {
                values[kj] -= aik * buffer[icount2 + int_row_size + k]; // aij = aij - aik*akj
              }
            }
            else {
              std::map<int, int>::iterator iter = inv_ids.find(uid);
              if (iter != inv_ids.end()) {
                int lid = iter->second;
                int kj = m_work[lid];
                if (kj != -1) {
                  values[kj] -= aik * buffer[icount2 + int_row_size + k]; // aij = aij - aik*akj
                }
              }
            }
          }

          for (int k = krow + 1; k < kcol[conn_row + 1]; ++k) {
            m_work[cols[k]] = -1;
          }
          for (int k = kcol[conn_row]; k < kcol[conn_row] + local_row_size[conn_row]; ++k) {
            m_work[cols[k]] = -1;
          }
        }
        icount += int_row_size + ext_row_size;
        icount2 += int_row_size + ext_row_size;
      }
    }
  }

 private:
  // clang-format off
  MatrixType&                                  m_matrix;
  MatrixDistribution&                          m_distribution ;
  std::vector< int >&                          m_work;
  const Alien::SimpleCSRInternal::CommInfo&    m_send_info;
  const Alien::SimpleCSRInternal::CommInfo&    m_recv_info;
#ifdef USE_SYCL_USM
  std::vector<ValueT*>                         m_send_lu_buffer;   
  std::vector<ValueT*>                         m_recv_lu_buffer;
  std::vector<int*>                            m_send_lu_ibuffer;   
  std::vector<int*>                            m_recv_lu_ibuffer;
#else
  std::vector<std::vector<ValueType>>          m_send_lu_buffer;
  std::vector<std::vector<ValueType>>          m_recv_lu_buffer;
  std::vector<std::vector<int>>                m_send_lu_ibuffer;
  std::vector<std::vector<int>>                m_recv_lu_ibuffer;
#endif

  UniqueArray< int >                           m_recv_connectivity_ids ;
  UniqueArray< int >                           m_recv_connectivity_krow ;
  UniqueArray< int >                           m_recv_connectivity_ids_ptr ;
  UniqueArray< std::map<int, int> >            m_mpi_ext_inv_ids ;

  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
  Arccore::ITraceMng*                          m_trace        = nullptr;
  // clang-format on
};

} // namespace Alien::SimpleCSRInternal
