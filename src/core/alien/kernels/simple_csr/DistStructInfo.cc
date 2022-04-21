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

/*
 * DistStructInfo.cc
 *
 *  Created on: Oct 1, 2010
 *      Author: gratienj
 */

#include <map>
#include <set>
#include <sstream>

#include "alien/core/block/VBlock.h"
#include "alien/core/block/VBlockOffsets.h"

#include "CSRStructInfo.h"
#include "DistStructInfo.h"
#include "SendRecvOp.h"

using namespace Arccore;
using namespace Arccore::MessagePassing;

namespace Alien::SimpleCSRInternal
{

void DistStructInfo::compute(Integer nproc, ConstArrayView<Integer> offset, Integer my_rank,
                             IMessagePassingMng* parallel_mng, const CSRStructInfo& profile,
                             ITraceMng* trace ALIEN_UNUSED_PARAM)
{
#ifdef DEBUG
  std::stringstream file("dd");
  file << my_rank;
  ofstream fout(file.str().c_str());
#endif
  std::vector<Integer> count(nproc);
  count.assign(nproc, 0);
  const Integer nrow = profile.getNRow();
  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  // ArrayView<Integer> cols = profile->getCols();
  m_cols.copy(profile.getCols());

  Integer local_offset = offset[my_rank];

  std::vector<std::map<Integer, Integer>> lids(nproc);
  std::vector<std::set<Integer>> gids(nproc);
  m_local_row_size.resize(nrow);
  for (Integer irow = 0; irow < nrow; ++irow) {
    Integer lrow_size = 0;
    for (Integer icol = row_offset[irow]; icol < row_offset[irow + 1]; ++icol) {
      Integer col_uid = m_cols[icol];
      Integer ip = domainId(nproc, offset, col_uid);
      // TODO: deal with ip == -1
      // fout<<"col("<<col_uid<<","<<ip<<");";
      if (ip == my_rank) {
        // turn col global id to local id
        m_cols[icol] = col_uid - local_offset;
        ++lrow_size;
      }
      else {
        /*
        std::map<Integer, Integer>::iterator iter = lids[ip].find(col_uid);
        if (iter == lids[ip].end()) {
          // compute local id of ghost id
          Integer col_lid = count[ip];
          lids[ip].insert(std::pair<Integer, Integer>(col_uid, col_lid));
          // cols[icol] = col_lid;
          ++count[ip];
        }*/
        gids[ip].insert(col_uid);
      }
    }
    // fout<<endl;
    m_local_row_size[irow] = lrow_size;
    if (lrow_size < row_offset[irow + 1] - row_offset[irow])
      ++m_interface_nrow;
  }
  // COMPUTE GHOST LOCAL ID PER NEIGH DOMAIN
  for (Integer ip = 0; ip < nproc; ++ip) {
    Integer lid = 0;
    auto& lids_ip = lids[ip];
    for (Integer uid : gids[ip]) {
      lids_ip.insert(std::make_pair(uid, lid));
      ++lid;
    }
  }
  {
    m_interface_rows.resize(m_interface_nrow);
    std::vector<Integer> col_offset(nproc + 1);
    col_offset[0] = nrow;
    for (Integer ip = 1; ip < nproc + 1; ++ip) {
      // col_offset[ip] = col_offset[ip - 1] + count[ip - 1];
      col_offset[ip] = col_offset[ip - 1] + gids[ip - 1].size();
    }
    std::size_t ghost_size = col_offset[nproc] - col_offset[0];
    m_recv_info.m_rank_ids.resize(ghost_size);
    m_recv_info.m_ids.resize(ghost_size);
    Integer icount = 0;
    Integer ghost_icount = 0;
    std::set<Integer> ghost_set;
    for (Integer irow = 0; irow < nrow; ++irow) {
      if (m_local_row_size[irow] < row_offset[irow + 1] - row_offset[irow]) {
        m_interface_rows[icount] = irow;
        m_interface_row_set.insert(irow);
        ++icount;
        for (Integer icol = row_offset[irow] + m_local_row_size[irow];
             icol < row_offset[irow + 1]; ++icol) {
          // compute local id of ghost col id
          Integer col_uid = m_cols[icol];
          Integer ip = domainId(nproc, offset, col_uid);
          m_cols[icol] = lids[ip][col_uid] + col_offset[ip];
          auto const& iter = ghost_set.insert(col_uid);
          if (iter.second) {
            m_recv_info.m_rank_ids[ghost_icount] = ip;
            m_recv_info.m_ids[ghost_icount] = m_cols[icol];
            ++ghost_icount;
          }
        }
      }
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  //
  // compute mailing list for mat mult communication operation
  //
  // INFO to recv from neighbour processors
  //
  Integer nb_neighbour = 0;
  Integer first_upper_neighb = 0;
  for (Integer ip = 0; ip < nproc; ++ip)
    // if (count[ip] > 0)
    //  ++nb_neighbour;
    if (gids[ip].size() > 0) {
      ++nb_neighbour;
      if (ip < my_rank)
        ++first_upper_neighb;
    }

  m_recv_info.m_num_neighbours = nb_neighbour;
  m_recv_info.m_first_upper_neighb = first_upper_neighb;
  m_recv_info.m_ranks.resize(nb_neighbour);
  m_recv_info.m_ids_offset.resize(nb_neighbour + 1);
  nb_neighbour = 0;
  m_recv_info.m_ids_offset[0] = nrow;
  for (Integer ip = 0; ip < nproc; ++ip)
    // if (count[ip] > 0) {
    if (gids[ip].size() > 0) {
      m_recv_info.m_ranks[nb_neighbour] = ip;
      // m_recv_info.m_ids_offset[nb_neighbour + 1] =
      // m_recv_info.m_ids_offset[nb_neighbour] + count[ip];
      m_recv_info.m_ids_offset[nb_neighbour + 1] =
      m_recv_info.m_ids_offset[nb_neighbour] + gids[ip].size();
      ++nb_neighbour;
    }
  m_ghost_nrow = m_recv_info.m_ids_offset[nb_neighbour] - nrow;
  m_recv_info.m_uids.resize(m_ghost_nrow);
  m_first_upper_ghost_index = m_recv_info.m_ids_offset[first_upper_neighb];
  // m_recv_info->printInfo(trace->info().file());

  //////////////////////////////////////////////////////////////////////////////
  //
  // compute mailing list for mat mult communication operation
  //
  // INFO to send to neighbour processor
  std::vector<std::vector<Integer>> send_ids(nproc);
  nb_neighbour = 0;
  Integer send_count = 0;
  std::vector<Integer> buffer;
  for (Integer ip = 0; ip < nproc; ++ip) {
    if (ip == my_rank) {
      Integer offset = 0;
      for (Integer ip2 = 0; ip2 < nproc; ++ip2) {
        if (ip2 != my_rank) {
          // Integer nids = lids[ip2].size();
          // Integer nids = count[ip2];
          Integer nids = gids[ip2].size();
          // trace->info()<<"SEND to "<<ip2<<" nids="<<nids;
          Arccore::MessagePassing::mpSend(
          parallel_mng, ConstArrayView<Integer>(1, &nids), ip2);
          if (nids > 0) {
            buffer.clear();
            buffer.resize(nids);
            // buffer.reserve(nids);
            std::map<Integer, Integer>::iterator iter = lids[ip2].begin();
            while (iter != lids[ip2].end()) {
              // buffer.push_back((*iter).first);
              buffer[(*iter).second] = (*iter).first;
              m_recv_info.m_uids[offset + (*iter).second] = (*iter).first;
              ++iter;
            }

            Arccore::MessagePassing::mpSend(
            parallel_mng, ConstArrayView<Integer>(nids, &buffer[0]), ip2);
            offset += nids;
          }
        }
      }
    }
    else {
      Integer nids = 0;
      Arccore::MessagePassing::mpReceive(parallel_mng, ArrayView<Integer>(1, &nids), ip);
      // trace->info()<<"RECV from "<<ip<<" nids="<<nids;
      if (nids > 0) {
        ++nb_neighbour;
        send_count += nids;
        send_ids[ip].resize(nids);
        Arccore::MessagePassing::mpReceive(
        parallel_mng, ArrayView<Integer>(nids, &send_ids[ip][0]), ip);
      }
    }
  }
  m_send_info.m_num_neighbours = nb_neighbour;
  m_send_info.m_ranks.resize(nb_neighbour);
  m_send_info.m_ids.resize(send_count);
  m_send_info.m_ids_offset.resize(nb_neighbour + 1);
  first_upper_neighb = 0;
  Integer icount = 0;
  nb_neighbour = 0;
  for (Integer ip = 0; ip < nproc; ++ip) {
    if (send_ids[ip].size() > 0) {
      m_send_info.m_ranks[nb_neighbour] = ip;
      m_send_info.m_ids_offset[nb_neighbour] = icount;
      ++nb_neighbour;
      if (ip < my_rank)
        ++first_upper_neighb;
      std::size_t nids = send_ids[ip].size();
      Integer* ids = &send_ids[ip][0];
      for (auto i = 0; i < nids; ++i) {
        m_send_info.m_ids[icount] = ids[i] - local_offset;
        // fout<<"SEND : "<<ids[i]<<" "<<ids[i] - local_offset<<endl;
        ++icount;
      }
    }
  }
  m_send_info.m_ids_offset[nb_neighbour] = icount;
  m_send_info.m_first_upper_neighb = first_upper_neighb;

  // m_send_info->printInfo(trace->info().file());
}

void DistStructInfo::compute(Integer nproc, ConstArrayView<Integer> offset, Integer my_rank,
                             IMessagePassingMng* parallel_mng, const CSRStructInfo& profile,
                             const VBlock* block_sizes, const MatrixDistribution& dist,
                             ITraceMng* trace ALIEN_UNUSED_PARAM)
{
  // alien_info([&] {cout() << "DistStructInfo::compute VBlock";}) ;
  std::vector<Integer> count(nproc);
  count.assign(nproc, 0);
  std::vector<Integer> block_count(nproc);
  block_count.assign(nproc, 0);

  Integer nrow = profile.getNRow();
  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  m_cols.copy(profile.getCols());

  VBlockImpl blocks(*block_sizes, dist);

  Integer block_nrow = 0;
  for (Integer i = 0; i < nrow; ++i) {
    block_nrow += blocks.sizeFromLocalIndex(i);
  }

  Integer local_offset = offset[my_rank];

  std::vector<std::map<Integer, Integer>> lids(nproc);
  std::vector<std::set<Integer>> gids(nproc);
  m_local_row_size.resize(nrow);
  for (Integer irow = 0; irow < nrow; ++irow) {
    Integer lrow_size = 0;
    // fout<<"ROW("<<irow<<")";
    for (Integer icol = row_offset[irow]; icol < row_offset[irow + 1]; ++icol) {
      Integer col_uid = m_cols[icol];
      Integer ip = domainId(nproc, offset, col_uid);
      // TODO: Deal with ip == -1
      // fout<<"col("<<col_uid<<","<<ip<<");";
      if (ip == my_rank) {
        // turn col global id to local id
        m_cols[icol] = col_uid - local_offset;
        ++lrow_size;
      }
      else {
        /*
        std::map<Integer, Integer>::iterator iter = lids[ip].find(col_uid);
        if (iter == lids[ip].end()) {
          // compute local id of ghost id
          Integer col_lid = count[ip];
          lids[ip].insert(std::pair<Integer, Integer>(col_uid, col_lid));
          // cols[icol] = col_lid;
          ++count[ip];
          block_count[ip] += block_sizes->size(col_uid);
        }*/
        auto value = gids[ip].insert(col_uid);
        if (value.second)
          block_count[ip] += block_sizes->size(col_uid);
      }
    }
    // fout<<endl;
    m_local_row_size[irow] = lrow_size;
    if (lrow_size < row_offset[irow + 1] - row_offset[irow])
      ++m_interface_nrow;
  }

  // COMPUTE GHOST LOCAL ID PER NEIGH DOMAIN
  for (int ip = 0; ip < nproc; ++ip) {
    int lid = 0;
    auto& lids_ip = lids[ip];
    for (int uid : gids[ip]) {
      lids_ip.insert(std::make_pair(uid, lid));
      ++lid;
    }
  }

  {
    // Il faut une map ordonn?e
    std::map<Integer, Integer> ghost_sizes;

    // Integer icount = 0;
    m_interface_rows.resize(m_interface_nrow);
    std::vector<Integer> col_offset(nproc + 1);
    col_offset[0] = nrow;
    for (Integer ip = 1; ip < nproc + 1; ++ip)
      // col_offset[ip] = col_offset[ip - 1] + count[ip - 1];
      col_offset[ip] = col_offset[ip - 1] + gids[ip - 1].size();
    std::size_t ghostsize = col_offset[nproc] - col_offset[0];
    m_recv_info.m_rank_ids.resize(ghostsize);
    m_recv_info.m_ids.resize(ghostsize);
    int icount = 0;
    int ghost_icount = 0;
    std::set<int> ghost_set;
    for (Integer irow = 0; irow < nrow; ++irow) {
      if (m_local_row_size[irow] < row_offset[irow + 1] - row_offset[irow]) {
        m_interface_rows[icount] = irow;
        m_interface_row_set.insert(irow);
        ++icount;
        // fout<<my_rank<<" Interface row : "<<irow << std::endl;
        for (Integer icol = row_offset[irow] + m_local_row_size[irow];
             icol < row_offset[irow + 1]; ++icol) {
          // compute local id of ghost col id
          Integer col_uid = m_cols[icol];
          Integer ip = domainId(nproc, offset, col_uid);
          m_cols[icol] = lids[ip][col_uid] + col_offset[ip];
          auto const& iter = ghost_set.insert(col_uid);
          if (iter.second) {
            m_recv_info.m_rank_ids[ghost_icount] = ip;
            m_recv_info.m_ids[ghost_icount] = m_cols[icol];
            ++ghost_icount;
          }
          // fout<<"COL["<<icol<<","<<col_uid<<"]="<<m_cols[icol]<<"
          // "<<col_offset[ip]<<endl; trace->info()
          // <<"COL["<<icol<<","<<col_uid<<"]="<<m_cols[icol]<<" "<<col_offset[ip];
          ghost_sizes[m_cols[icol]] = block_sizes->size(col_uid);
          // block_sizes.setGhostLocalId(col_uid,m_cols[icol]);
        }
      }
    }

    ConstArrayView<Integer> local_sizes = blocks.sizeOfLocalIndex();
    ConstArrayView<Integer> local_offsets = blocks.offsetOfLocalIndex();

    ALIEN_ASSERT(
    (local_offsets.size() == local_sizes.size() + 1), ("sizes are different"));

    const Integer size = local_sizes.size();
    const Integer ghost_size = ghost_sizes.size();

    m_block_sizes.resize(size + ghost_size);
    m_block_offsets.resize(size + ghost_size);

    for (Integer i = 0; i < size; ++i) {
      m_block_sizes[i] = local_sizes[i];
      m_block_offsets[i] = local_offsets[i];
    }

    for (std::map<Integer, Integer>::const_iterator it = ghost_sizes.begin();
         it != ghost_sizes.end(); ++it) {
      const Integer lid = it->first;
      m_block_sizes[lid] = it->second;
      m_block_offsets[lid] = m_block_offsets[lid - 1] + m_block_sizes[lid - 1];
    }
  }

  //   for(Integer i = 0; i < m_block_sizes.size(); ++i) {
  //     trace->info() << "block_sizes[" << i << "] = " << m_block_sizes[i];
  //     trace->info() << "block_offsets[" << i << "] = " << m_block_offsets[i];
  //   }

  //////////////////////////////////////////////////////////////////////////////
  //
  // compute mailing list for mat mult communication operation
  //
  // INFO to recv from neighbour processors
  //
  Integer nb_neighbour = 0;
  // for (Integer ip = 0; ip < nproc; ++ip)
  //  if (count[ip] > 0)
  //    ++nb_neighbour;
  int first_upper_neighb = 0;
  for (int ip = 0; ip < nproc; ++ip)
    if (gids[ip].size() > 0) {
      ++nb_neighbour;
      if (ip < my_rank)
        ++first_upper_neighb;
    }
  m_recv_info.m_num_neighbours = nb_neighbour;
  m_recv_info.m_first_upper_neighb = first_upper_neighb;
  m_recv_info.m_ranks.resize(nb_neighbour);
  m_recv_info.m_ids_offset.resize(nb_neighbour + 1);
  m_recv_info.m_block_ids_offset.resize(nb_neighbour + 1);
  nb_neighbour = 0;
  m_recv_info.m_ids_offset[0] = nrow;
  m_recv_info.m_block_ids_offset[0] = block_nrow;
  for (Integer ip = 0; ip < nproc; ++ip)
    // if (count[ip] > 0) {
    if (gids[ip].size() > 0) {
      m_recv_info.m_ranks[nb_neighbour] = ip;
      // m_recv_info.m_ids_offset[nb_neighbour + 1] =
      // m_recv_info.m_ids_offset[nb_neighbour] + count[ip];
      m_recv_info.m_ids_offset[nb_neighbour + 1] =
      m_recv_info.m_ids_offset[nb_neighbour] + gids[ip].size();
      m_recv_info.m_block_ids_offset[nb_neighbour + 1] =
      m_recv_info.m_block_ids_offset[nb_neighbour] + block_count[ip];
      ++nb_neighbour;
    }
  m_ghost_nrow = m_recv_info.m_ids_offset[nb_neighbour] - nrow;
  m_recv_info.m_uids.resize(m_ghost_nrow);
  m_first_upper_ghost_index = m_recv_info.m_ids_offset[first_upper_neighb];

  // m_recv_info.printInfo(recv_fout);

  //////////////////////////////////////////////////////////////////////////////
  //
  // compute mailing list for mat mult communication operation
  //
  // INFO to send to neighbour processor
  std::vector<std::vector<Integer>> send_ids(nproc);
  nb_neighbour = 0;
  Integer send_count = 0;
  std::vector<Integer> buffer;
  for (Integer ip = 0; ip < nproc; ++ip) {
    if (ip == my_rank) {
      Integer offset = 0;
      for (Integer ip2 = 0; ip2 < nproc; ++ip2) {
        if (ip2 != my_rank) {
          // Integer nids = lids[ip2].size();
          // Integer nids = count[ip2];
          Integer nids = gids[ip2].size();
          // trace->info()<<"SEND to "<<ip2<<" nids="<<nids;
          Arccore::MessagePassing::mpSend(
          parallel_mng, ConstArrayView<Integer>(1, &nids), ip2);
          if (nids > 0) {
            buffer.clear();
            buffer.resize(nids);
            // buffer.reserve(nids);
            std::map<Integer, Integer>::iterator iter = lids[ip2].begin();
            while (iter != lids[ip2].end()) {
              // buffer.push_back((*iter).first);
              buffer[(*iter).second] = (*iter).first;
              m_recv_info.m_uids[offset + (*iter).second] = (*iter).first;
              ++iter;
            }

            Arccore::MessagePassing::mpSend(
            parallel_mng, ConstArrayView<Integer>(nids, &buffer[0]), ip2);
            offset += nids;
          }
        }
      }
    }
    else {
      Integer nids = 0;
      Arccore::MessagePassing::mpReceive(parallel_mng, ArrayView<Integer>(1, &nids), ip);
      // trace->info()<<"RECV from "<<ip<<" nids="<<nids;
      if (nids > 0) {
        ++nb_neighbour;
        send_count += nids;
        send_ids[ip].resize(nids);
        Arccore::MessagePassing::mpReceive(
        parallel_mng, ArrayView<Integer>(nids, &send_ids[ip][0]), ip);
      }
    }
  }
  m_send_info.m_num_neighbours = nb_neighbour;
  m_send_info.m_ranks.resize(nb_neighbour);
  m_send_info.m_ids.resize(send_count);
  m_send_info.m_ids_offset.resize(nb_neighbour + 1);
  m_send_info.m_block_ids_offset.resize(nb_neighbour + 1);
  Integer icount = 0, block_icount = 0;
  nb_neighbour = 0;
  for (Integer ip = 0; ip < nproc; ++ip) {
    if (send_ids[ip].size() > 0) {
      m_send_info.m_ranks[nb_neighbour] = ip;
      m_send_info.m_ids_offset[nb_neighbour] = icount;
      m_send_info.m_block_ids_offset[nb_neighbour] = block_icount;
      ++nb_neighbour;
      if (ip < my_rank)
        ++first_upper_neighb;
      Integer nids = send_ids[ip].size();
      Integer* ids = &send_ids[ip][0];
      for (Integer i = 0; i < nids; ++i) {
        m_send_info.m_ids[icount] = ids[i] - local_offset;
        ++icount;
        block_icount += m_block_sizes[ids[i] - local_offset];
      }
    }
  }
  m_send_info.m_ids_offset[nb_neighbour] = icount;
  m_send_info.m_block_ids_offset[nb_neighbour] = block_icount;

  // m_send_info.printInfo(send_fout);
}

void DistStructInfo::copy(const DistStructInfo& src)
{
  m_local_row_size.copy(src.m_local_row_size);
  m_ghost_nrow = src.m_ghost_nrow;
  m_interface_nrow = src.m_interface_nrow;
  m_first_upper_ghost_index = src.m_first_upper_ghost_index;

  m_interface_rows.copy(src.m_interface_rows);
  m_interface_row_set = src.m_interface_row_set;
  m_cols.copy(src.m_cols);

  m_upper_diag_offset.copy(src.m_upper_diag_offset);
  m_block_sizes.copy(src.m_block_sizes);
  m_block_offsets.copy(src.m_block_offsets);

  m_recv_info.copy(src.m_recv_info);
  m_send_info.copy(src.m_send_info);
}

} // namespace Alien::SimpleCSRInternal
