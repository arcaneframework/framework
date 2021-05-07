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
 * RedistributorVector.cc
 *
 *  Created on: 27 juil. 2016
 *      Author: chevalic
 */
#include "RedistributorVector.h"

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/utils/time_stamp/TimestampObserver.h>

#include "RedistributorBackEnd.h"
#include "RedistributorCommPlan.h"

namespace Alien
{
using namespace Arccore;

RedistributorVector::RedistributorVector(const MultiVectorImpl* src_impl)
: IVectorImpl(src_impl, AlgebraTraits<BackEnd::tag::redistributor>::name())
, m_super_pm(nullptr)
, m_tgt_impl(nullptr)
{
  m_super_pm = distribution().parallelMng();
}

void RedistributorVector::init(const VectorDistribution& dist ALIEN_UNUSED_PARAM,
                               const bool need_allocate ALIEN_UNUSED_PARAM)
{
  return;
}

void RedistributorVector::clear()
{
  // m_tgt_impl.reset(nullptr);
  // m_tgt_dist.reset(nullptr);
}

void RedistributorVector::updateSuperPM(MultiVectorImpl* tgt_impl ALIEN_UNUSED_PARAM,
                                        const RedistributorCommPlan* commPlan ALIEN_UNUSED_PARAM)
{
  // TODO Throw exception
  /*
  if (!m_tgt_impl || m_tgt_impl->distribution().parallelMng() !=
  commPlan->tgtParallelMng()) return; redistributeBack(tgt_impl);
  */
}

std::shared_ptr<MultiVectorImpl>
RedistributorVector::updateTargetPM(const RedistributorCommPlan* commPlan)
{
  if (m_tgt_impl && m_tgt_impl->distribution().parallelMng() == commPlan->tgtParallelMng().get())
    return m_tgt_impl;

  m_tgt_dist.reset(new VectorDistribution(commPlan->distribution()));
  m_tgt_impl.reset(new MultiVectorImpl(space().clone(), m_tgt_dist));

  m_tgt_impl->addObserver(std::make_shared<TimestampObserver>(*this));

  const VectorDistribution& src_dist = distribution();
  m_distributor.reset(new SimpleCSRVectorRedistributor(commPlan, src_dist));
  return redistribute();
}

std::shared_ptr<MultiVectorImpl>
RedistributorVector::redistribute()
{
  auto& vec_src = m_multi_impl->get<BackEnd::tag::simplecsr>();
  auto& vec_tgt = m_tgt_impl->get<BackEnd::tag::simplecsr>(true);
  m_distributor->distribute(vec_src, vec_tgt);
  return m_tgt_impl;
}

void RedistributorVector::redistributeBack(SimpleCSRVector<Real>& vec_tgt) const
{
  auto& vec_src = m_tgt_impl->get<BackEnd::tag::simplecsr>();
  m_distributor->distributeBack(vec_src, vec_tgt);
}

} // namespace Alien
