// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceBuilder.cc                                           (C) 2000-2023 */
/*                                                                           */
/* Classe utilitaire pour instantier un service.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ServiceBuilder.h"

#include "arcane/core/ICaseMng.h"
#include "arcane/core/CaseOptions.h"
#include "arcane/core/internal/ICaseMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ReferenceCounter<ICaseOptions> ServiceBuilderWithOptionsBase::
_buildCaseOptions(const AxlOptionsBuilder::Document& options_doc) const
{
  ReferenceCounter<ICaseOptions> co = CaseOptions::createDynamic(m_case_mng, options_doc);
  return co;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IApplication* ServiceBuilderWithOptionsBase::
_application() const
{
  return m_case_mng->application();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ServiceBuilderWithOptionsBase::
_readOptions(ICaseOptions* opt) const
{
  m_case_mng->_internalImpl()->internalReadOneOption(opt, true);
  m_case_mng->_internalImpl()->internalReadOneOption(opt, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

