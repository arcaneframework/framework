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

#include <alien/kernels/redistributor/Utils.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/utils/Precomp.h>

namespace Alien
{

class RedistributorCommPlan;

class SimpleCSRVectorRedistributor
{
 public:
  SimpleCSRVectorRedistributor(
  const RedistributorCommPlan* commPlan, const VectorDistribution& src_dist);
  virtual ~SimpleCSRVectorRedistributor();

  void distribute(const SimpleCSRVector<Real>& src, SimpleCSRVector<Real>& tgt);
  void distributeBack(const SimpleCSRVector<Real>& src, SimpleCSRVector<Real>& tgt);

  Int32 rcvSize() const { return m_rcv_offset[m_rcv_offset.size() - 1]; }

  Int32 rcvBackSize() const { return m_snd_offset[m_snd_offset.size() - 1]; }

 private:
  void _computeCommPlan(const VectorDistribution& src_dist);

 private:
  const RedistributorCommPlan* m_comm_plan;
  IMessagePassingMng* m_pm_super;
  IMessagePassingMng* m_pm_dst;

  // Prepare communication buffer
  UniqueArray<Int32> m_snd_offset;
  UniqueArray<Int32> m_rcv_offset;
};

} // namespace Alien
