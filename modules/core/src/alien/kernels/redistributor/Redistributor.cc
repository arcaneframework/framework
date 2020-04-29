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
 * Redistributor.cpp
 *
 *  Created on: 28 juil. 2016
 *      Author: chevalic
 */

#include "Redistributor.h"

#include <arccore/message_passing/Messages.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>

#include "RedistributorBackEnd.h"
#include "RedistributorMatrix.h"
#include "RedistributorVector.h"

using namespace Arccore::MessagePassing;
using namespace Arccore;

namespace Alien {

Redistributor::Redistributor(
    int globalSize, Arccore::MessagePassing::IMessagePassingMng* super, bool keep)
: m_super_pm(super)
{
  m_distributor.reset(new RedistributorCommPlan(
      globalSize, m_super_pm, Arccore::MessagePassing::mpSplit(super, keep)));
}

Redistributor::~Redistributor() {}

std::shared_ptr<MultiMatrixImpl>
Redistributor::redistribute(MultiMatrixImpl* mat)
{
  auto& red_mat = mat->get<BackEnd::tag::redistributor>(true);
  return red_mat.updateTargetPM(m_distributor.get());
}

std::shared_ptr<MultiVectorImpl>
Redistributor::redistribute(MultiVectorImpl* vect)
{
  auto& red_vect = vect->get<BackEnd::tag::redistributor>(false);
  return red_vect.updateTargetPM(m_distributor.get());
}

void
Redistributor::redistributeBack(MultiVectorImpl* vect ALIEN_UNUSED_PARAM)
{
  /*
    auto& dst_vect = vect->get<BackEnd::tag::redistributor>(true);
    dst_vect.updateSuperPM(vect, m_distributor.get());
  */
}

const RedistributorCommPlan*
Redistributor::commPlan() const
{
  return m_distributor.get();
}

} // namespace Alien
