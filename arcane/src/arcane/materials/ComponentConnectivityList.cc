// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentConnectivityList.cc                                (C) 2000-2023 */
/*                                                                           */
/* Gestion des listes de connectivité des milieux et matériaux.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/ComponentConnectivityList.h"

#include "arcane/materials/internal/MeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentConnectivityList::
ComponentConnectivityList(MeshMaterialMng* mm)
: TraceAccessor(mm->traceMng())
, m_material_mng(mm)
, m_cell_nb_environment(VariableBuildInfo(mm->meshHandle(), mm->name() + "_CellNbEnvironment2"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentConnectivityList::
endCreate(bool is_continue)
{
  if (!is_continue)
    m_cell_nb_environment.fill(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
