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

  Directory dir(m_path);

  String dir_path = dir.path();

  String rogn = dir_path.substring(0, dir_path.len()-1);
  info() << rogn;


  // if (name.startsWith("/")) {
  //   name = "/" + name;
  // }

  // StringUniqueArray string_splited;
  // name.split(string_splited, '/');

  // if (string_splited.size() > 1) {
  //   std::optional<Integer> proc_id = string_splited.span().findFirst("proc_id");
  //   if (proc_id) {
  //     string_splited[proc_id.value()] = String::fromNumber(mesh()->parallelMng()->commRank());
  //     only_once = false;
  //   }
  //   else {
  //     only_once = true;
  //   }

  //   std::optional<Integer> num_procs = string_splited.span().findFirst("num_procs");
  //   if (num_procs) {
  //     string_splited[num_procs.value()] = String::fromNumber(mesh()->parallelMng()->commSize());
  //   }
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
