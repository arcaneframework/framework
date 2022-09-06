// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DirectSubDomainExecutionContext.cc                          (C) 2000-2022 */
/*                                                                           */
/* Contexte d'exécution directe avec création d'un sous-domaine.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/DirectSubDomainExecutionContext.h"

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class DirectSubDomainExecutionContext::Impl
{
 public:
  Impl(ISubDomain* sd) : m_sub_domain(sd){}
  ISubDomain* m_sub_domain;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DirectSubDomainExecutionContext::
DirectSubDomainExecutionContext(ISubDomain* sd)
: m_p(new Impl(sd))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DirectSubDomainExecutionContext::
~DirectSubDomainExecutionContext()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* DirectSubDomainExecutionContext::
subDomain() const
{
  return m_p->m_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
