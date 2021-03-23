// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionsArcane.h                                         (C) 2000-2006 */
/*                                                                           */
/* Options principales de Arcane.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/CaseOptionsMain.h"
#include "arcane/CaseOptionBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionsMain::
CaseOptionsMain(ICaseMng* cm)
: CaseOptions(cm,String("main"))
, doTimeHistory(CaseOptionBuildInfo(configList(),String("do-time-history"),XmlNode(0),String("true"),1,1),String())
, writeHistoryPeriod(CaseOptionBuildInfo(configList(),String("write-history-period"),XmlNode(0),String("0"),1,1),String())
{
  doTimeHistory.addAlternativeNodeName(String("fr"),String("avec-historique"));
  writeHistoryPeriod.addAlternativeNodeName(String("fr"),String("periode-ecriture-historique"));
  addAlternativeNodeName(String("fr"),String("maitre"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionsMain::
~CaseOptionsMain()
{
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
