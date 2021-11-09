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
 * RedistributorMatrix.cc
 *
 *  Created on: 27 juil. 2016
 *      Author: chevalic
 */

#include "RedistributorMatrix.h"

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/dok/DoKBackEnd.h>
#include <alien/kernels/dok/DoKMatrixT.h>

#include "RedistributorBackEnd.h"
#include "RedistributorCommPlan.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;
using namespace Arccore::MessagePassing;

RedistributorMatrix::RedistributorMatrix(const MultiMatrixImpl* src_impl)
: IMatrixImpl(src_impl, AlgebraTraits<BackEnd::tag::redistributor>::name())
, m_super_pm(nullptr)
, m_tgt_impl(nullptr)
{
  // Cannot call our "distribution()" from a constructor.
  m_super_pm = IMatrixImpl::distribution().parallelMng();
}

void RedistributorMatrix::clear()
{
  // m_tgt_impl.reset(nullptr);
  // m_tgt_dist.reset(nullptr);
}

void RedistributorMatrix::setSuperPM(IMessagePassingMng* pm)
{
  m_super_pm = pm;
}

std::shared_ptr<MultiMatrixImpl>
RedistributorMatrix::updateTargetPM(const RedistributorCommPlan* commPlan)
{
  if (m_tgt_impl && m_tgt_impl->distribution().parallelMng() == commPlan->tgtParallelMng().get())
    return m_tgt_impl;

  const MatrixDistribution& src_dist = distribution();
  m_tgt_dist.reset(new MatrixDistribution(
  src_dist.globalRowSize(), src_dist.globalColSize(), commPlan->tgtParallelMng()));
  m_tgt_impl.reset(
  new MultiMatrixImpl(rowSpace().clone(), colSpace().clone(), m_tgt_dist));

  // Now, we have to exchange data, using DoK representation.
  m_distributor.reset(new DoKDistributor(commPlan));
  return redistribute();
}

std::shared_ptr<MultiMatrixImpl>
RedistributorMatrix::redistribute()
{
  auto& mat_src = m_multi_impl->get<BackEnd::tag::DoK>();
  auto& mat_tgt = m_tgt_impl->get<BackEnd::tag::DoK>(true);
  m_distributor->distribute(mat_src, mat_tgt);
  return m_tgt_impl;
}

} // namespace Alien
