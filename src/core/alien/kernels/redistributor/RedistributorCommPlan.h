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

#include <alien/kernels/redistributor/RedistributorBackEnd.h>
#include <memory>

#include <alien/distribution/VectorDistribution.h>

namespace Alien
{

class ALIEN_EXPORT RedistributorCommPlan
{
 public:
  RedistributorCommPlan(
  int globalSize, IMessagePassingMng* super_pm, IMessagePassingMng* tgt_pm);
  virtual ~RedistributorCommPlan();

  std::shared_ptr<IMessagePassingMng> tgtParallelMng() const;
  IMessagePassingMng* superParallelMng() const;

  const VectorDistribution& distribution() const;

  ConstArrayView<Int32> tgtDist() const;

  [[nodiscard]] Int32 procNum(Int32) const;

 private:
  void _buildTgtDist();

  IMessagePassingMng* m_super_pm;
  IMessagePassingMng* m_tgt_pm;
  std::unique_ptr<VectorDistribution> m_tgt_distribution; //! Distribution in the target pm
  UniqueArray<Int32> m_proc_num; //! Array for converting ranks from super to dst
  UniqueArray<Int32> m_tgt_dist; //! This array will contain the target distribution, relative to the
  //! super communication manager.
};

} // namespace Alien
