// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptions.cc                                              (C) 2000-2023 */
/*                                                                           */
/* Gestion des options du jeu de données.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionComplexValue.h"

#include "arcane/core/ICaseOptions.h"
#include "arcane/core/internal/ICaseOptionListInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionComplexValue::
CaseOptionComplexValue(ICaseOptionsMulti* opt,ICaseOptionList* clist,const XmlNode& parent_elem)
: m_config_list(ICaseOptionListInternal::create(clist,opt->toCaseOptions(),parent_elem,clist->isOptional(),true))
, m_element(parent_elem)
{
  opt->addChild(_configList());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionComplexValue::
~CaseOptionComplexValue()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
