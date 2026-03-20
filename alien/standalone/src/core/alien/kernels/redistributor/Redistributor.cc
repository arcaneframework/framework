// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "Redistributor.h"

#include <arccore/message_passing/Messages.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>

#include <memory>

#include "RedistributorBackEnd.h"
#include "RedistributorMatrix.h"
#include "RedistributorVector.h"

using namespace Arccore::MessagePassing;
using namespace Arccore;

namespace Alien
{

Redistributor::Redistributor(
int globalSize, Arccore::MessagePassing::IMessagePassingMng* super, Arccore::MessagePassing::IMessagePassingMng* target, Method method)
: m_super_pm(super)
, m_distributor(std::make_unique<RedistributorCommPlan>(globalSize, m_super_pm, target))
, m_method(method)
{
}

std::shared_ptr<MultiMatrixImpl>
Redistributor::redistribute(MultiMatrixImpl* mat)
{
  auto& red_mat = mat->get<BackEnd::tag::redistributor>(true);
  if (m_method == Method::csr) {
    red_mat.useCSRRedistributor();
  }
  return red_mat.updateTargetPM(m_distributor.get());
}

std::shared_ptr<MultiVectorImpl>
Redistributor::redistribute(MultiVectorImpl* vect)
{
  auto& red_vect = vect->get<BackEnd::tag::redistributor>(false);
  return red_vect.updateTargetPM(m_distributor.get());
}

void Redistributor::redistributeBack(MultiVectorImpl* vect ALIEN_UNUSED_PARAM)
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
