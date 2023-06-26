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

#include <vector>

#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/utils/Precomp.h>
#include <alien/utils/Trace.h>

#include <arccore/message_passing/Messages.h>
#include <arccore/message_passing/Request.h>

namespace Arccore
{
class ITraceMng;
}

namespace Alien::SimpleCSRInternal
{

struct CommProperty
{
  typedef enum
  {
    Synch,
    ASynch
  } ePolicyType;
};

struct CommInfo
{
  CommInfo() {}

  void printInfo(std::ostream& fout) const
  {
    fout << "Num of neighbours : " << m_num_neighbours << std::endl;
    fout << "Ranks : ";
    for (Integer i = 0; i < m_num_neighbours; ++i)
      fout << m_ranks[i] << " ";
    fout << std::endl;
    if (m_ids.size()) {
      for (Integer i = 0; i < m_num_neighbours; ++i) {
        fout << "List[" << i << "] : " << m_ids_offset[i] << " " << m_ids_offset[i + 1]
             << std::endl;
        for (Integer k = m_ids_offset[i]; k < m_ids_offset[i + 1]; ++k)
          fout << "        id[" << k << "]=" << m_ids[k] << std::endl;
        if (m_block_ids_offset.size())
          fout << "Block list[" << i << "] : " << m_block_ids_offset[i] << " "
               << m_block_ids_offset[i + 1] << std::endl;
      }
    }
    else {
      if (m_block_ids_offset.size()) {
        for (Integer i = 0; i < m_num_neighbours + 1; ++i)
          fout << "offset[" << i << "]=" << m_ids_offset[i]
               << ",block=" << m_block_ids_offset[i] << std::endl;
      }
      else {
        for (Integer i = 0; i < m_num_neighbours + 1; ++i) {
          fout << "offset[" << i << "]" << m_ids_offset[i] << std::endl;
        }
      }
    }
  }
  inline Arccore::Integer getDomainId(Arccore::Integer id) const
  {
    if (id < m_ids_offset[0])
      return -1;
    for (Arccore::Integer ineighb = 0; ineighb < m_num_neighbours; ++ineighb) {
      if (id < m_ids_offset[ineighb + 1])
        return m_ranks[ineighb];
    }
    return -2;
  }

  inline Arccore::Integer getNeighbId(Arccore::Integer id) const
  {
    if (id < m_ids_offset[0])
      return -1;
    for (Arccore::Integer ineighb = 0; ineighb < m_num_neighbours; ++ineighb) {
      if (id < m_ids_offset[ineighb + 1])
        return ineighb;
    }
    return -2;
  }

  inline Arccore::Integer getRankNeighbId(Arccore::Integer rank) const
  {
    for (Arccore::Integer ineighb = 0; ineighb < m_num_neighbours; ++ineighb) {
      if (m_ranks[ineighb] == rank)
        return ineighb;
    }
    return -1;
  }

  Arccore::Integer getLocalId(Arccore::Integer ineighb, Arccore::Integer id) const
  {
    for (Arccore::Integer k = m_ids_offset[ineighb]; k < m_ids_offset[ineighb + 1]; ++k) {
      if (m_ids[k] == id)
        return k - m_ids_offset[ineighb];
    }
    return -1;
  }

  Arccore::Integer getLocalIdFromUid(Arccore::Integer ineighb, Arccore::Integer uid) const
  {
    for (Arccore::Integer k = m_ids_offset[ineighb]; k < m_ids_offset[ineighb + 1]; ++k) {
      if (m_uids[k - m_ids_offset[0]] == uid)
        return k;
    }
    return -1;
  }

  Arccore::Integer m_num_neighbours = 0;
  Arccore::Integer m_first_upper_neighb = 0;
  Arccore::UniqueArray<Arccore::Integer> m_ranks;
  Arccore::UniqueArray<Arccore::Integer> m_ids;
  Arccore::UniqueArray<Arccore::Integer> m_uids;
  Arccore::UniqueArray<Arccore::Integer> m_rank_ids;
  Arccore::UniqueArray<Arccore::Integer> m_ids_offset;
  Arccore::UniqueArray<Arccore::Integer> m_block_ids_offset;

  void copy(const CommInfo& commInfo)
  {
    m_num_neighbours = commInfo.m_num_neighbours;
    m_first_upper_neighb = commInfo.m_first_upper_neighb;

    m_ranks.copy(commInfo.m_ranks);
    m_ids.copy(commInfo.m_ids);
    m_uids.copy(commInfo.m_uids);
    m_rank_ids.copy(commInfo.m_rank_ids);
    m_ids_offset.copy(commInfo.m_ids_offset);
    m_block_ids_offset.copy(commInfo.m_block_ids_offset);
  }
};

class IASynchOp
{
 public:
  IASynchOp() {}

  virtual ~IASynchOp() {}
  virtual void start(bool insitu = true) = 0;
  virtual void end(bool insitu = true) = 0;
};

template <typename ValueT>
class SendRecvOp : public IASynchOp
{
 public:
  SendRecvOp(const ValueT* send_buffer,
             const CommInfo& send_info,
             CommProperty::ePolicyType send_policy,
             ValueT* recv_buffer,
             const CommInfo& recv_info,
             CommProperty::ePolicyType recv_policy,
             IMessagePassingMng* mng, Arccore::ITraceMng* trace_mng,
             Integer unknowns_num = 1)
  : m_is_variable_block(false)
  , m_send_buffer(send_buffer)
  , m_send_info(send_info)
  , m_send_policy(send_policy)
  , m_recv_buffer(recv_buffer)
  , m_recv_info(recv_info)
  , m_recv_policy(recv_policy)
  , m_parallel_mng(mng)
  , m_trace(trace_mng)
  , m_unknowns_num(unknowns_num)
  {}

  SendRecvOp(const ValueT* send_buffer,
             const CommInfo& send_info,
             CommProperty::ePolicyType send_policy,
             ValueT* recv_buffer,
             const CommInfo& recv_info,
             CommProperty::ePolicyType recv_policy,
             IMessagePassingMng* mng,
             Arccore::ITraceMng* trace_mng,
             ConstArrayView<Integer> block_sizes,
             ConstArrayView<Integer> block_offsets)
  : m_is_variable_block(true)
  , m_send_buffer(send_buffer)
  , m_send_info(send_info)
  , m_send_policy(send_policy)
  , m_recv_buffer(recv_buffer)
  , m_recv_info(recv_info)
  , m_recv_policy(recv_policy)
  , m_parallel_mng(mng)
  , m_trace(trace_mng)
  , m_block_sizes(block_sizes)
  , m_block_offsets(block_offsets)
  {}

  void start(bool insitu = true)
  {
    if (m_is_variable_block)
      _startBlock(insitu);
    else
      _start(insitu);
  }

  void end(bool insitu = true)
  {
    if (m_is_variable_block)
      _endBlock(insitu);
    else
      _end(insitu);
  }

  void recv(bool insitu = true)
  {
    ValueT* rbuffer = m_recv_buffer;
    if (m_recv_info.m_ids.size() && !insitu) {
      int size = m_recv_info.m_ids_offset[m_recv_info.m_num_neighbours];
      m_rbuffer.resize(size * m_unknowns_num);
      rbuffer = &m_rbuffer[0];
    }
    for (int i = 0; i < m_recv_info.m_num_neighbours; ++i) {
      int off = m_recv_info.m_ids_offset[i];
      int size = m_recv_info.m_ids_offset[i + 1] - off;
      ValueT* ptr = rbuffer + off * m_unknowns_num;
      int rank = m_recv_info.m_ranks[i];
      Arccore::MessagePassing::mpReceive(m_parallel_mng, ArrayView<ValueT>(size * m_unknowns_num, ptr), rank);
    }
    if (m_recv_info.m_ids.size() && !insitu) {
      int size = m_recv_info.m_ids_offset[m_recv_info.m_num_neighbours] -
      m_recv_info.m_ids_offset[0];
      if (m_unknowns_num == 1)
        for (int i = 0; i < size; ++i)
          m_recv_buffer[m_recv_info.m_ids[i]] = m_rbuffer[i];
      else
        for (int i = 0; i < size; ++i)
          for (std::size_t ui = 0; ui < m_unknowns_num; ++ui)
            m_recv_buffer[m_recv_info.m_ids[i] * m_unknowns_num + ui] = m_rbuffer[i * m_unknowns_num + ui];
    }
  }

  void lowerRecv(bool insitu = true)
  {
    ValueT* rbuffer = m_recv_buffer;
    if (m_recv_info.m_ids.size() && !insitu) {
      int size = m_recv_info.m_ids_offset[m_recv_info.m_first_upper_neighb];
      m_rbuffer.resize(size * m_unknowns_num);
      rbuffer = &m_rbuffer[0];
    }
    for (int i = 0; i < m_recv_info.m_first_upper_neighb; ++i) {
      int off = m_recv_info.m_ids_offset[i];
      int size = m_recv_info.m_ids_offset[i + 1] - off;
      ValueT* ptr = rbuffer + off * m_unknowns_num;
      int rank = m_recv_info.m_ranks[i];
      Arccore::MessagePassing::mpReceive(m_parallel_mng, ArrayView<ValueT>(size * m_unknowns_num, ptr), rank);
    }
    if (m_recv_info.m_ids.size() && !insitu) {
      int size = m_recv_info.m_ids_offset[m_recv_info.m_first_upper_neighb] -
      m_recv_info.m_ids_offset[0];
      if (m_unknowns_num == 1)
        for (int i = 0; i < size; ++i)
          m_recv_buffer[m_recv_info.m_ids[i]] = m_rbuffer[i];
      else
        for (int i = 0; i < size; ++i)
          for (std::size_t ui = 0; ui < m_unknowns_num; ++ui)
            m_recv_buffer[m_recv_info.m_ids[i] * m_unknowns_num + ui] = m_rbuffer[i * m_unknowns_num + ui];
    }
  }

  void upperRecv(bool insitu = true)
  {
    ValueT* rbuffer = m_recv_buffer;
    int size = 0;
    if (m_recv_info.m_ids.size() && !insitu) {
      size = m_recv_info.m_ids_offset[m_recv_info.m_num_neighbours] - m_recv_info.m_ids_offset[m_recv_info.m_first_upper_neighb];
      m_rbuffer.resize(size * m_unknowns_num);
      rbuffer = &m_rbuffer[0];
    }
    //for(int i=m_recv_info.m_first_upper_neighb;i<m_recv_info.m_num_neighbours;++i)
    for (int i = m_recv_info.m_num_neighbours - 1; i > m_recv_info.m_first_upper_neighb - 1; --i) {
      int off = m_recv_info.m_ids_offset[i];
      int size = m_recv_info.m_ids_offset[i + 1] - off;
      ValueT* ptr = rbuffer + off * m_unknowns_num;
      int rank = m_recv_info.m_ranks[i];
      Arccore::MessagePassing::mpReceive(m_parallel_mng, ArrayView<ValueT>(size * m_unknowns_num, ptr), rank);
    }
    if (m_recv_info.m_ids.size() && !insitu) {
      int size = m_recv_info.m_ids_offset[m_recv_info.m_num_neighbours] -
      m_recv_info.m_ids_offset[m_recv_info.m_first_upper_neighb];
      if (m_unknowns_num == 1)
        for (int i = 0; i < size; ++i)
          m_recv_buffer[m_recv_info.m_ids[i]] = m_rbuffer[i];
      else
        for (int i = 0; i < size; ++i)
          for (std::size_t ui = 0; ui < m_unknowns_num; ++ui)
            m_recv_buffer[m_recv_info.m_ids[i] * m_unknowns_num + ui] = m_rbuffer[i * m_unknowns_num + ui];
    }
  }

  void send()
  {
    ValueT const* sbuffer = m_send_buffer;
    if (m_send_info.m_ids.size()) {
      int size = m_send_info.m_ids_offset[m_send_info.m_num_neighbours] -
      m_send_info.m_ids_offset[0];
      m_sbuffer.resize(size * m_unknowns_num);
      if (m_unknowns_num == 1)
        for (int i = 0; i < size; ++i) {
          m_sbuffer[i] = m_send_buffer[m_send_info.m_ids[i]];
          //m_trace->info()<<"SEND"<<m_sbuffer[i];
        }
      else
        for (int i = 0; i < size; ++i)
          for (int ui = 0; ui < m_unknowns_num; ++ui)
            m_sbuffer[i * m_unknowns_num + ui] = m_send_buffer[m_send_info.m_ids[i] * m_unknowns_num + ui];
      sbuffer = &m_sbuffer[0];
    }
    for (int i = 0; i < m_send_info.m_num_neighbours; ++i) {
      int off = m_send_info.m_ids_offset[i];
      int size = m_send_info.m_ids_offset[i + 1] - off;
      ValueT const* ptr = sbuffer + off * m_unknowns_num;
      int rank = m_send_info.m_ranks[i];
      Arccore::MessagePassing::mpSend(m_parallel_mng, ConstArrayView<ValueT>(size * m_unknowns_num, ptr), rank);
    }
  }

  void lowerSend()
  {
    ValueT const* sbuffer = m_send_buffer;
    if (m_send_info.m_ids.size()) {
      int size = m_send_info.m_ids_offset[m_send_info.m_first_upper_neighb] -
      m_send_info.m_ids_offset[0];
      m_sbuffer.resize(size * m_unknowns_num);
      if (m_unknowns_num == 1)
        for (int i = 0; i < size; ++i) {
          m_sbuffer[i] = m_send_buffer[m_send_info.m_ids[i]];
          //m_trace->info()<<"SEND"<<m_sbuffer[i];
        }
      else
        for (int i = 0; i < size; ++i)
          for (std::size_t ui = 0; ui < m_unknowns_num; ++ui)
            m_sbuffer[i * m_unknowns_num + ui] = m_send_buffer[m_send_info.m_ids[i] * m_unknowns_num + ui];
      sbuffer = &m_sbuffer[0];
    }
    //for(int i=0;i<m_send_info.m_first_upper_neighb;++i)
    for (int i = m_send_info.m_first_upper_neighb - 1; i > -1; --i) {
      int off = m_send_info.m_ids_offset[i];
      int size = m_send_info.m_ids_offset[i + 1] - off;
      ValueT const* ptr = sbuffer + off * m_unknowns_num;
      int rank = m_send_info.m_ranks[i];
      Arccore::MessagePassing::mpSend(m_parallel_mng, ConstArrayView<ValueT>(size * m_unknowns_num, ptr), rank);
    }
  }

  void upperSend()
  {
    ValueT const* sbuffer = nullptr;
    if (m_send_info.m_ids.size()) {
      int offset = m_send_info.m_ids_offset[m_send_info.m_first_upper_neighb] - m_send_info.m_ids_offset[0];
      int size = m_send_info.m_ids_offset[m_send_info.m_num_neighbours] - m_send_info.m_ids_offset[m_send_info.m_first_upper_neighb];
      m_sbuffer.resize(size * m_unknowns_num);
      if (m_unknowns_num == 1)
        for (int i = 0; i < size; ++i) {
          m_sbuffer[i] = m_send_buffer[m_send_info.m_ids[offset + i]];
        }
      else
        for (int i = 0; i < size; ++i)
          for (std::size_t ui = 0; ui < m_unknowns_num; ++ui)
            m_sbuffer[i * m_unknowns_num + ui] = m_send_buffer[m_send_info.m_ids[offset + i] * m_unknowns_num + ui];
      sbuffer = &m_sbuffer[0];
    }
    else
      sbuffer = m_send_buffer + m_send_info.m_ids_offset[m_send_info.m_first_upper_neighb] * m_unknowns_num;
    ValueT const* ptr = sbuffer;
    for (int i = m_send_info.m_first_upper_neighb; i < m_send_info.m_num_neighbours; ++i) {
      int size = m_send_info.m_ids_offset[i + 1] - m_send_info.m_ids_offset[i];
      int rank = m_send_info.m_ranks[i];
      Arccore::MessagePassing::mpSend(m_parallel_mng, ConstArrayView<ValueT>(size * m_unknowns_num, ptr), rank);
      ptr += size * m_unknowns_num;
    }
  }

 private:
  void _start(bool insitu)
  {
    if (m_recv_policy == CommProperty::ASynch) {
      m_recv_request.resize(m_recv_info.m_num_neighbours);
      ValueT* rbuffer = nullptr;
      if (m_recv_info.m_ids.size() && !insitu) {
        Integer size = m_recv_info.m_ids_offset[m_recv_info.m_num_neighbours] - m_recv_info.m_ids_offset[0];
        m_rbuffer.resize(size * m_unknowns_num);
        rbuffer = &m_rbuffer[0];
      }
      else
        rbuffer = m_recv_buffer;
      for (Integer i = 0; i < m_recv_info.m_num_neighbours; ++i) {
        Integer off = m_recv_info.m_ids_offset[i];
        Integer size = m_recv_info.m_ids_offset[i + 1] - off;
        ValueT* ptr = rbuffer + off * m_unknowns_num;
        Integer rank = m_recv_info.m_ranks[i];
        m_recv_request[i] = Arccore::MessagePassing::mpReceive(
        m_parallel_mng, ArrayView<ValueT>(size * m_unknowns_num, ptr), rank, false);
      }
    }
    if (m_send_policy == CommProperty::ASynch)
      m_send_request.resize(m_send_info.m_num_neighbours);
    ValueT const* sbuffer = m_send_buffer;
    if (m_send_info.m_ids.size()) {
      Integer size = m_send_info.m_ids_offset[m_send_info.m_num_neighbours] - m_send_info.m_ids_offset[0];
      m_sbuffer.resize(size * m_unknowns_num);
      if (m_unknowns_num == 1)
        for (Integer i = 0; i < size; ++i) {
          m_sbuffer[i] = m_send_buffer[m_send_info.m_ids[i]];
        }
      else
        for (Integer i = 0; i < size; ++i)
          for (Integer ui = 0; ui < m_unknowns_num; ++ui)
            m_sbuffer[i * m_unknowns_num + ui] =
            m_send_buffer[m_send_info.m_ids[i] * m_unknowns_num + ui];
      sbuffer = &m_sbuffer[0];
    }
    for (Integer i = 0; i < m_send_info.m_num_neighbours; ++i) {
      Integer off = m_send_info.m_ids_offset[i];
      Integer size = m_send_info.m_ids_offset[i + 1] - off;
      ValueT const* ptr = sbuffer + off * m_unknowns_num;
      Integer rank = m_send_info.m_ranks[i];
      if (m_send_policy == CommProperty::ASynch)
        m_send_request[i] = Arccore::MessagePassing::mpSend(m_parallel_mng,
                                                            ConstArrayView<ValueT>(size * m_unknowns_num, ptr), rank, false);
      else
        Arccore::MessagePassing::mpSend(
        m_parallel_mng, ConstArrayView<ValueT>(size * m_unknowns_num, ptr), rank);
    }
  }

  void _end(bool insitu)
  {
    if (m_recv_policy == CommProperty::ASynch)
      Arccore::MessagePassing::mpWaitAll(m_parallel_mng, m_recv_request);
    else {
      ValueT* rbuffer = m_recv_buffer;
      if (m_recv_info.m_ids.size() && !insitu) {
        Arccore::Integer size = m_recv_info.m_ids_offset[m_recv_info.m_num_neighbours];
        m_rbuffer.resize(size * m_unknowns_num);
        rbuffer = &m_rbuffer[0];
      }
      for (Integer i = 0; i < m_recv_info.m_num_neighbours; ++i) {
        Integer off = m_recv_info.m_ids_offset[i];
        Integer size = m_recv_info.m_ids_offset[i + 1] - off;
        ValueT* ptr = rbuffer + off * m_unknowns_num;
        Integer rank = m_recv_info.m_ranks[i];
        Arccore::MessagePassing::mpReceive(
        m_parallel_mng, ArrayView<ValueT>(size * m_unknowns_num, ptr), rank);
      }
      if (m_recv_info.m_ids.size() && !insitu) {
        Integer size = m_recv_info.m_ids_offset[m_recv_info.m_num_neighbours] - m_recv_info.m_ids_offset[0];
        if (m_unknowns_num == 1)
          for (Integer i = 0; i < size; ++i)
            m_recv_buffer[m_recv_info.m_ids[i]] = m_rbuffer[i];
        else
          for (Integer i = 0; i < size; ++i)
            for (Integer ui = 0; ui < m_unknowns_num; ++ui)
              m_recv_buffer[m_recv_info.m_ids[i] * m_unknowns_num + ui] =
              m_rbuffer[i * m_unknowns_num + ui];
      }
    }
    if (m_send_policy == CommProperty::ASynch)
      Arccore::MessagePassing::mpWaitAll(m_parallel_mng, m_send_request);
  }

  void _startBlock(bool insitu)
  {
    // alien_info([&] {cout() << "StartBlock "<<insitu<<" send pol"<<m_send_policy;});
    if (m_recv_policy == CommProperty::ASynch) {
      m_recv_request.resize(m_recv_info.m_num_neighbours);
      ValueT* rbuffer = nullptr;
      if (m_recv_info.m_ids.size() && !insitu) {
        Integer size = m_recv_info.m_block_ids_offset[m_recv_info.m_num_neighbours] - m_recv_info.m_block_ids_offset[0];
        m_rbuffer.resize(size);
        rbuffer = &m_rbuffer[0];
      }
      else
        rbuffer = m_recv_buffer;
      // alien_info([&] {cout() << "RecvInfo Nb Neighb :
      // "<<m_recv_info.m_num_neighbours;}) ;
      for (Integer i = 0; i < m_recv_info.m_num_neighbours; ++i) {
        Integer off = m_recv_info.m_block_ids_offset[i];
        Integer size = m_recv_info.m_block_ids_offset[i + 1] - off;
        ValueT* ptr = rbuffer + off;
        Integer rank = m_recv_info.m_ranks[i];
        m_recv_request[i] = Arccore::MessagePassing::mpReceive(
        m_parallel_mng, ArrayView<ValueT>(size, ptr), rank, false);
      }
    }
    if (m_send_policy == CommProperty::ASynch)
      m_send_request.resize(m_send_info.m_num_neighbours);
    ValueT const* sbuffer = m_send_buffer;
    if (m_send_info.m_ids.size()) {
      {
        Integer size = m_send_info.m_block_ids_offset[m_send_info.m_num_neighbours] - m_send_info.m_block_ids_offset[0];
        m_sbuffer.resize(size);
      }
      Integer size = m_send_info.m_ids_offset[m_send_info.m_num_neighbours] - m_send_info.m_ids_offset[0];
      Integer offset = 0;
      for (Integer i = 0; i < size; ++i) {
        Integer id = m_send_info.m_ids[i];
        Integer block_size = m_block_sizes[id];
        Integer block_offset = m_block_offsets[id];
        for (Integer ui = 0; ui < block_size; ++ui)
          m_sbuffer[offset + ui] = m_send_buffer[block_offset + ui];
        offset += block_size;
      }
      // ARCANE_ASSERT((offset ==
      // m_send_info.m_block_ids_offset[m_send_info.m_num_neighbours] -
      // m_send_info.m_block_ids_offset[0]),("size error"));
      sbuffer = &m_sbuffer[0];
    }
    for (Integer i = 0; i < m_send_info.m_num_neighbours; ++i) {
      Integer off = m_send_info.m_block_ids_offset[i];
      Integer size = m_send_info.m_block_ids_offset[i + 1] - off;
      ValueT const* ptr = sbuffer + off;
      Integer rank = m_send_info.m_ranks[i];
      if (m_send_policy == CommProperty::ASynch)
        m_send_request[i] = Arccore::MessagePassing::mpSend(
        m_parallel_mng, ConstArrayView<ValueT>(size, ptr), rank, false);
      else
        Arccore::MessagePassing::mpSend(
        m_parallel_mng, ConstArrayView<ValueT>(size, ptr), rank);
    }
  }

  void _endBlock(bool insitu)
  {
    // alien_info([&] {cout() << "EndBlock "<<insitu<<" recv pol="<<m_recv_policy;});
    if (m_recv_policy == CommProperty::ASynch) {
      Arccore::MessagePassing::mpWaitAll(m_parallel_mng, m_recv_request);
    }
    else {
      ValueT* rbuffer = m_recv_buffer;
      if (m_recv_info.m_ids.size() && !insitu) {
        Arccore::Integer size =
        m_recv_info.m_block_ids_offset[m_recv_info.m_num_neighbours];
        m_rbuffer.resize(size);
        rbuffer = &m_rbuffer[0];
      }
      for (Integer i = 0; i < m_recv_info.m_num_neighbours; ++i) {
        Integer off = m_recv_info.m_block_ids_offset[i];
        Integer size = m_recv_info.m_block_ids_offset[i + 1] - off;
        ValueT* ptr = rbuffer + off;
        Integer rank = m_recv_info.m_ranks[i];
        Arccore::MessagePassing::mpReceive(
        m_parallel_mng, ArrayView<ValueT>(size, ptr), rank);
      }
      if (m_recv_info.m_ids.size() && !insitu) {
        Arccore::Integer size = m_recv_info.m_ids_offset[m_recv_info.m_num_neighbours] - m_recv_info.m_ids_offset[0];
        Integer offset = 0;
        for (Integer i = 0; i < size; ++i) {
          Integer id = m_recv_info.m_ids[i];
          Integer block_size = m_block_sizes[id];
          Integer block_offset = m_block_offsets[id];
          for (Integer ui = 0; ui < block_size; ++ui)
            m_recv_buffer[block_offset + ui] = m_rbuffer[offset + ui];
          offset += block_size;
        }
        // ARCANE_ASSERT((offset =
        // m_recv_info.m_block_ids_offset[m_recv_info.m_num_neighbours] -
        // m_recv_info.m_block_ids_offset[0]),("size error"));
      }
    }
    if (m_send_policy == CommProperty::ASynch)
      Arccore::MessagePassing::mpWaitAll(m_parallel_mng, m_send_request);
  }

 private:
  const bool m_is_variable_block = false;
  const ValueT* m_send_buffer = nullptr;
  const CommInfo& m_send_info;
  CommProperty::ePolicyType m_send_policy;
  ValueT* m_recv_buffer = nullptr;
  const CommInfo& m_recv_info;
  CommProperty::ePolicyType m_recv_policy;
  Arccore::MessagePassing::IMessagePassingMng* m_parallel_mng = nullptr;
  Arccore::ITraceMng* m_trace = nullptr;
  Arccore::Integer m_unknowns_num = 0;
  std::vector<ValueT> m_rbuffer;
  std::vector<ValueT> m_sbuffer;
  Arccore::UniqueArray<Arccore::MessagePassing::Request> m_recv_request;
  Arccore::UniqueArray<Arccore::MessagePassing::Request> m_send_request;
  Arccore::ConstArrayView<Arccore::Integer> m_block_sizes;
  Arccore::ConstArrayView<Arccore::Integer> m_block_offsets;
};

} // namespace Alien::SimpleCSRInternal
