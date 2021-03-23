// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DirectExecutionContext.cc                                   (C) 2000-2021 */
/*                                                                           */
/* Contexte d'exécution directe.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/DirectExecutionContext.h"

#include "arcane/utils/String.h"
#include "arcane/launcher/IDirectExecutionContext.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DirectExecutionContext::
DirectExecutionContext(IDirectExecutionContext* ctx)
: m_p(ctx)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* DirectExecutionContext::
createSequentialSubDomain()
{
  return m_p->createSequentialSubDomain(String());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* DirectExecutionContext::
createSequentialSubDomain(const String& case_file_name)
{
  return m_p->createSequentialSubDomain(case_file_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
