﻿/// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <alien/utils/Precomp.h>
#include <alien/distribution/MatrixDistribution.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/redistributor/RedistributorCommPlan.h>
#include <arccore/message_passing/PointToPointMessageInfo.h>
#include <arccore/base/Span.h>

namespace Alien
{

class ALIEN_EXPORT SimpleCSRDistributor
{
 public:
  SimpleCSRDistributor(const RedistributorCommPlan* commPlan, const VectorDistribution& source_distribution,
                       const Alien::SimpleCSRInternal::CSRStructInfo* src_profile);

  virtual ~SimpleCSRDistributor() = default;

  template <typename NumT>
  void distribute(const SimpleCSRMatrix<NumT>& src, SimpleCSRMatrix<NumT>& dst);

  template <typename NumT>
  void distribute(const SimpleCSRVector<NumT>& src, SimpleCSRVector<NumT>& dst);

  [[nodiscard]] std::shared_ptr<const Alien::SimpleCSRInternal::CSRStructInfo> getDstProfile() const
  {
    return m_dst_profile;
  }

 private:
  template <typename T>
  std::optional<T> _owner(const std::vector<T>& offset, T global_row_id);
  template <typename T>
  void _distribute(const int bb, const T* src, T* dst);
  template <typename T>
  void _resizeBuffers(const int bb);
  void _finishExchange();
  std::optional<int> _dstMe(int) const;

  struct CommInfo
  {
   public:
    std::vector<Integer> m_row_list;
    std::size_t m_n_item = 0;
    std::vector<uint64_t> m_buffer;
    Arccore::MessagePassing::PointToPointMessageInfo m_message_info;
    Arccore::MessagePassing::Request m_request;
  };

  const RedistributorCommPlan* m_comm_plan = nullptr;

  const Alien::SimpleCSRInternal::CSRStructInfo* m_src_profile = nullptr;
  std::shared_ptr<Alien::SimpleCSRInternal::CSRStructInfo> m_dst_profile;

  // map<processor id,<num item,row list>>
  std::map<int, CommInfo> m_send_comm_info;
  std::map<int, CommInfo> m_recv_comm_info;

  // list of rows that can directly transferred from source matrix to destination matrix
  std::vector<std::pair<Integer, Integer>> m_src2dst_row_list;
};

} // namespace Alien
