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

#include <alien/core/impl/IVectorImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/redistribution/SimpleCSRVectorRedistributor.h>
#include <alien/utils/Precomp.h>

namespace Alien
{

// Forward declarations
class RedistributorCommPlan;

class RedistributorVector : public IVectorImpl
{
 public:
  RedistributorVector(const MultiVectorImpl* src_impl);
  virtual ~RedistributorVector() {}

  void init(const VectorDistribution& dist, const bool need_allocate) override;

  //! Clear data
  void clear() override;

  std::shared_ptr<MultiVectorImpl> updateTargetPM(const RedistributorCommPlan* commPlan);
  void updateSuperPM(MultiVectorImpl* tgt_impl, const RedistributorCommPlan* commPlan);

  std::shared_ptr<MultiVectorImpl> redistribute();
  void redistributeBack(SimpleCSRVector<Real>& vec_tgt) const;

 private:
  const IMessagePassingMng* m_super_pm;
  std::shared_ptr<MultiVectorImpl> m_tgt_impl;
  std::shared_ptr<VectorDistribution> m_tgt_dist;
  std::unique_ptr<SimpleCSRVectorRedistributor> m_distributor;
};

} // namespace Alien
