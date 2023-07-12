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

#include "SimpleCSRVectorRedistributor.h"

#include <alien/kernels/redistributor/RedistributorCommPlan.h>
#include <alien/utils/Precomp.h>

namespace Alien
{

using namespace Arccore;

SimpleCSRVectorRedistributor::SimpleCSRVectorRedistributor(const RedistributorCommPlan* commPlan, const VectorDistribution& src_dist)
: m_comm_plan(commPlan)
, m_pm_super(m_comm_plan->superParallelMng())
, m_pm_dst(m_comm_plan->tgtParallelMng().get())
{
  this->_computeCommPlan(src_dist);
}

SimpleCSRVectorRedistributor::~SimpleCSRVectorRedistributor() {}

void SimpleCSRVectorRedistributor::_computeCommPlan(const VectorDistribution& src_dist)
{
  Int32 super_comm_size = m_pm_super->commSize();

  // This array will contain the target distribution, relative to the super communication
  // manager.
  ConstArrayView<Int32> tgt_dist = m_comm_plan->tgtDist();

  const Integer localSize = src_dist.localSize();
  const Integer offset = src_dist.offset();

  // Prepare communication buffer
  UniqueArray<Int32> snd_rows(localSize, 0);

  for (Integer i = 0; i < localSize; ++i)
    snd_rows[i] = i + offset;

  Int32 p = 0;
  // I prefer to pass p and rowid in an explicit way !
  auto is_mine = [&](Int32 p, Int32 rowid) {
    return (tgt_dist[p] <= rowid) && (rowid < tgt_dist[p + 1]);
  };

  m_snd_offset.resize(super_comm_size + 1);
  m_snd_offset.fill(-1);
  m_snd_offset[0] = 0;
  for (Integer i = 0; i < localSize; i++) {
    Int32 row_id = snd_rows[i];
    while ((!is_mine(p, row_id)) && p < super_comm_size) {
      p++;
      m_snd_offset[p] = i;
    }
  }
  for (int p2 = p; p2 < super_comm_size; p2++)
    m_snd_offset[p2 + 1] = localSize;

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

void SimpleCSRVectorRedistributor::distribute(
const SimpleCSRVector<Real>& src, SimpleCSRVector<Real>& tgt)
{
  // Copy values to a send buffer, in case src and dst are the same matrix.
  UniqueArray<Real> snd_values = src.fullValues(); // TODO: avoid this copy
  UniqueArray<Real> rcv_values(this->rcvSize());

  Alien::RedistributionTools::exchange(m_pm_super, snd_values.constView(),
                                       m_snd_offset.constView(), rcv_values.view(), m_rcv_offset.constView());

  ArrayView<Real> values = tgt.fullValues();
  for (Integer i = 0; i < values.size(); ++i)
    values[i] = rcv_values[i];
}

void SimpleCSRVectorRedistributor::distributeBack(
const SimpleCSRVector<Real>& src, SimpleCSRVector<Real>& tgt)
{
  // Copy values to a send buffer, in case src and dst are the same matrix.
  UniqueArray<Real> snd_values = src.fullValues(); // TODO: avoid this copy
  UniqueArray<Real> rcv_values(this->rcvBackSize());

  Alien::RedistributionTools::exchange(m_pm_super, snd_values.constView(),
                                       m_rcv_offset.constView(), rcv_values.view(), m_snd_offset.constView());

  ArrayView<Real> values = tgt.fullValues();
  for (Integer i = 0; i < values.size(); ++i)
    values[i] = rcv_values[i];
}

} // namespace Alien
