// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvComparatorService.cc                                   (C) 2000-2022 */
/*                                                                           */
/* Service permettant de construire et de sortir un tableau au formet csv.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleCsvComparatorService.h"

#include <arcane/Directory.h>
#include <arcane/IMesh.h>
#include <arcane/IParallelMng.h>

#include <optional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvComparatorService::
addSimpleTableOutputEntry(ISimpleTableOutput* ptr_sto)
{
  ARCANE_CHECK_PTR(ptr_sto);
  m_iSTO = ptr_sto;
}

void SimpleCsvComparatorService::
readSimpleTableOutputEntry()
{
  m_name_tab = m_iSTO->name();
  m_path = m_iSTO->path();
  m_path.endsWith("/");
}
void SimpleCsvComparatorService::
editRefFileEntry(String path, String name, bool no_edit_path)
{
  
}
bool SimpleCsvComparatorService::
writeRefFile(Integer only_proc)
{
  return false;
}
bool SimpleCsvComparatorService::
writeRefFile(String path, Integer only_proc)
{
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
