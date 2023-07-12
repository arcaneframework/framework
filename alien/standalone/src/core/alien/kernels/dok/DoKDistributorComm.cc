/*

Copyright 2020 IFPEN-CEA

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "DoKDistributorComm.h"

#include <alien/kernels/redistributor/RedistributorCommPlan.h>

namespace Alien
{
using namespace Arccore;

DoKDistributorComm::DoKDistributorComm(const RedistributorCommPlan* commPlan)
: m_comm_plan(commPlan)
, m_pm_super(m_comm_plan->superParallelMng())
, m_pm_dst(m_comm_plan->tgtParallelMng().get())
{}

void DoKDistributorComm::computeCommPlan(const Arccore::Span<Int32> base)
{
  Int32 super_comm_size = m_pm_super->commSize();

  // This array will contain the target distribution, relative to the super communication
  // manager.
  ConstArrayView<Int32> tgt_dist = m_comm_plan->tgtDist();

  // Now, all processors know the target data distribution.
  auto size = base.size();

  Int32 p = 0;
  // I prefer to pass p and rowid in an explicit way !
  auto is_mine = [&](Int32 p, Int32 rowid) {
    return (tgt_dist[p] <= rowid) && (rowid < tgt_dist[p + 1]);
  };

  m_snd_offset.resize(super_comm_size + 1);
  m_snd_offset.fill(-1);
  m_snd_offset[0] = 0;
  int i = 0;
  for (auto row_id : base) {
    while (p < super_comm_size // Check if p is valid
           && (!is_mine(p, row_id))) {
      p++;
      m_snd_offset[p] = i;
    }
    i++;
  }
  for (int p2 = p; p2 < super_comm_size; p2++) {
    m_snd_offset[p2 + 1] = size;
  }

  UniqueArray<Int32> snd_count(super_comm_size);
  Alien::RedistributionTools::computeCounts(m_snd_offset.constView(), snd_count.view());
  m_rcv_offset.resize(super_comm_size + 1);
  UniqueArray<Int32> rcv_count(super_comm_size);

  Arccore::MessagePassing::mpAllToAll(m_pm_super, snd_count, rcv_count, 1);

  m_rcv_offset[0] = 0;
  for (p = 0; p < super_comm_size; ++p) {
    m_rcv_offset[p + 1] = m_rcv_offset[p] + rcv_count[p];
  }
}

void DoKDistributorComm::computeCommPlan(IReverseIndexer* rev_index)
{
  // Now, all processors know the target data distribution.
  Int32 size = 0;
  if (rev_index) {
    size = rev_index->size();
  }

  // Prepare communication buffer
  UniqueArray<Int32> snd_rows(size, 0);
  UniqueArray<Int32> snd_cols(size, 0);

  for (IReverseIndexer::Offset offset = 0; offset < size; ++offset) {
    auto ij = (*rev_index)[offset].value();
    snd_rows[offset] = ij.first;
    snd_cols[offset] = ij.second;
  }

  this->computeCommPlan(snd_rows);

  // Arccore is not smart enough to resize reception buffers...
  m_rcv_rows.resize(m_rcv_offset[m_pm_super->commSize()]);
  m_rcv_cols.resize(m_rcv_offset[m_pm_super->commSize()]);

  Alien::RedistributionTools::exchange(m_pm_super, snd_rows.constView(),
                                       m_snd_offset.constView(), m_rcv_rows.view(), m_rcv_offset.constView());
  Alien::RedistributionTools::exchange(m_pm_super, snd_cols.constView(),
                                       m_snd_offset.constView(), m_rcv_cols.view(), m_rcv_offset.constView());
}

} // namespace Alien
