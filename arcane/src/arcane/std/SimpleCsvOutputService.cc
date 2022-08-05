// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleCsvOutputService.cc                                   (C) 2000-2022 */
/*                                                                           */
/* Service permettant de construire et de sortir un tableau au formet csv.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/SimpleCsvOutputService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
init()
{
  if (m_with_option && options()->getTableName() != "") {
    return m_simple_table_output_mng.init(options()->getTableName());
  }
  return m_simple_table_output_mng.init();
}

bool SimpleCsvOutputService::
init(const String& table_name)
{
  if (m_with_option) {
    return m_simple_table_output_mng.init(table_name, options()->getTableDir());
  }
  return m_simple_table_output_mng.init(table_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
writeFile(const String& directory, Integer process_id)
{
  m_simple_table_output_mng.setOutputDirectory(directory);
  return m_simple_table_output_mng.writeFile(process_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
