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

#include <arccore/base/FatalErrorException.h>
#include <arccore/base/TraceInfo.h>

#include <alien/kernels/redistributor/Utils.h>

#include <alien/kernels/dok/DoKReverseIndexer.h>

namespace Alien
{

class RedistributorCommPlan;

class ALIEN_EXPORT DoKDistributorComm
{
 public:
  typedef IReverseIndexer::Index Index;

 public:
  /*!
   * @brief Construct a DoKDistributorComm object over parallelMng:
   *  - super is the master parallelMng
   *  - src is the source parallelMng, sub-communicator of super
   *  - dst is the destination parallelMng, sub-communicator of super.
   */
  explicit DoKDistributorComm(const RedistributorCommPlan* commPlan);

  virtual ~DoKDistributorComm() = default;

  DoKDistributorComm(const DoKDistributorComm& src) = delete;
  DoKDistributorComm& operator=(const DoKDistributorComm& src) = delete;
  DoKDistributorComm(const DoKDistributorComm&& src) = delete;
  DoKDistributorComm& operator=(DoKDistributorComm&& src) = delete;

  void computeCommPlan(IReverseIndexer* rev_index);

  void computeCommPlan(Arccore::Span<Int32>);

  template <typename T>
  void exchange(ConstArrayView<T> snd, ArrayView<T> rcv)
  {
    Alien::RedistributionTools::exchange(
    m_pm_super, snd, m_snd_offset.constView(), rcv, m_rcv_offset.constView());
  }

  Int32 rcvSize() const { return m_rcv_offset[m_rcv_offset.size() - 1]; }

  Index getCoordinates(Int32 offset) const
  {
    if ((offset < 0) || (m_rcv_rows.size() <= offset))
      throw FatalErrorException("Invalid offset in DoKDistributorComm");
    return { m_rcv_rows[offset], m_rcv_cols[offset] };
  }

 private:
  const RedistributorCommPlan* m_comm_plan;
  IMessagePassingMng* m_pm_super;
  IMessagePassingMng* m_pm_dst;

  // Prepare communication buffer
  UniqueArray<Int32> m_snd_offset;
  UniqueArray<Int32> m_rcv_offset;
  UniqueArray<Int32> m_rcv_rows;
  UniqueArray<Int32> m_rcv_cols;
};

} // namespace Alien
