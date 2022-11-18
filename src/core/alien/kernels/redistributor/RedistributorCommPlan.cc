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

#include "RedistributorCommPlan.h"

#include <arccore/message_passing/Messages.h>

#include <alien/utils/Precomp.h>

namespace Alien
{

using namespace Arccore;
using namespace Arccore::MessagePassing;

RedistributorCommPlan::RedistributorCommPlan(
int globalSize, IMessagePassingMng* super_pm, IMessagePassingMng* target_pm)
: m_super_pm(super_pm)
, m_tgt_pm(target_pm)
, m_proc_num(super_pm->commSize() + 1, -1) // Why + 1 ?
, m_tgt_dist(super_pm->commSize() + 1, -1)
{
  m_tgt_distribution.reset(new VectorDistribution(globalSize, m_tgt_pm));
  Int32 tgt_rank = -1;
  if (m_tgt_pm) {
    tgt_rank = m_tgt_pm->commRank();
    m_proc_num.resize(m_tgt_pm->commSize());
  }

  // TODO avoid communication if m_pm_dst is m_pm_super
  UniqueArray<Int32> reverse_proc_num(m_super_pm->commSize());
  Arccore::MessagePassing::mpAllGather(
  m_super_pm, ConstArrayView<Int32>(1, &tgt_rank), reverse_proc_num);

  for (Int32 i = 0; i < (Int32)reverse_proc_num.size(); ++i) {
    Int32 rank = reverse_proc_num[i];
    if (rank < 0)
      continue;
    m_proc_num[rank] = i;
  }
  this->_buildTgtDist();
}

RedistributorCommPlan::~RedistributorCommPlan() {}

std::shared_ptr<IMessagePassingMng>
RedistributorCommPlan::tgtParallelMng() const
{
  return m_tgt_distribution->sharedParallelMng();
}

IMessagePassingMng*
RedistributorCommPlan::superParallelMng() const
{
  return m_super_pm;
}

const VectorDistribution&
RedistributorCommPlan::distribution() const
{
  return *m_tgt_distribution;
}

ConstArrayView<Int32>
RedistributorCommPlan::tgtDist() const
{
  return m_tgt_dist.view();
}

Int32 RedistributorCommPlan::procNum(Int32 proc) const
{
  return m_proc_num[proc];
}

void RedistributorCommPlan::_buildTgtDist()
{
  Int32 super_comm_size = m_super_pm->commSize();
  Int32 super_rank = 0;
  // Not all procs know the target communicator.
  // Thus we need to communicate the distribution to these processors.
  if (m_tgt_pm) {
    Int32 dst_comm_size = m_tgt_pm->commSize();
    for (Int32 p = 0; p < dst_comm_size; p++) {
      m_tgt_dist[m_proc_num[p]] = m_tgt_distribution->offset(p);
    }
    m_tgt_dist[super_comm_size] = m_tgt_distribution->globalSize();

    // Fill other processes.
    if (m_tgt_dist[0] < 0)
      m_tgt_dist[0] = 0;
    for (Int32 p = super_comm_size; p > 0; --p) {
      if (m_tgt_dist[p] < 0) {
        m_tgt_dist[p] = m_tgt_dist[p + 1];
      }
    }
    super_rank = m_super_pm->commRank();
  }

  // This communication can be avoided if we store a root in the ctor.
  Int32 root = Arccore::MessagePassing::mpAllReduce(
  m_super_pm, Arccore::MessagePassing::ReduceMax, super_rank);

  // Broadcast from root to all processes.
  Arccore::MessagePassing::mpBroadcast(m_super_pm, m_tgt_dist, root);
}

} // namespace Alien
