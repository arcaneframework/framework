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

#include <alien/core/impl/IMatrixImpl.h>
#include <alien/kernels/dok/DoKDistributor.h>
#include <alien/kernels/simple_csr/redistribution/SimpleCSRDistributor.h>
#include <alien/utils/Precomp.h>

namespace Alien
{

// Forward declarations
class RedistributorCommPlan;

class ALIEN_EXPORT RedistributorMatrix : public IMatrixImpl
{
 public:
  explicit RedistributorMatrix(const MultiMatrixImpl* src_impl, bool use_dok = true);
  ~RedistributorMatrix() = default;

  RedistributorMatrix(const RedistributorMatrix& src) = delete;
  RedistributorMatrix(RedistributorMatrix&& src) = delete;
  RedistributorMatrix& operator=(const RedistributorMatrix& src) = delete;
  RedistributorMatrix& operator=(RedistributorMatrix&& src) = delete;

  //! Demande la lib�ration des donn�es
  void clear() override;

  void useCSRRedistributor();
  std::shared_ptr<MultiMatrixImpl> updateTargetPM(const RedistributorCommPlan* commPlan);
  void setSuperPM(IMessagePassingMng* pm);

  std::shared_ptr<MultiMatrixImpl> redistribute();

 private:
  const IMessagePassingMng* m_super_pm;
  std::shared_ptr<MultiMatrixImpl> m_tgt_impl;
  std::shared_ptr<MatrixDistribution> m_tgt_dist;
  std::unique_ptr<DoKDistributor> m_distributor;
  bool m_use_dok = true;
  std::unique_ptr<SimpleCSRDistributor> m_simple_csr_distibutor;
};

} // namespace Alien
