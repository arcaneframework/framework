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
    return init(options()->getTableName());
  }
  return init("Table_P@proc_id@");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
init(const String& table_name)
{
  if (m_with_option) {
    return init(table_name, options()->getTableDir());
  }
  return init(table_name, "");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
init(const String& table_name, const String& directory_name)
{
  if (!m_is_already_init) {
    m_is_already_init = true;

    m_internal = makeRef(new SimpleTableInternal(mesh()->parallelMng()));
    m_simple_csv_reader_writer = makeRef(new SimpleCsvReaderWriter(m_internal));

    m_simple_table_internal_mng.setInternal(m_internal);
    m_simple_table_output_mng.setReaderWriter(m_simple_csv_reader_writer);
  }
  return m_simple_table_output_mng.init(subDomain()->exportDirectory(), table_name, directory_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimpleCsvOutputService::
clear()
{
  if (m_is_already_init) {
    m_simple_table_internal_mng.clearInternal();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool SimpleCsvOutputService::
writeFile(const String& directory, Integer rank)
{
  m_simple_table_output_mng.setOutputDirectory(directory);
  return m_simple_table_output_mng.writeFile(rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_SIMPLECSVOUTPUT(SimpleCsvOutput, SimpleCsvOutputService);

ARCANE_REGISTER_SERVICE(SimpleCsvOutputService,
                        ServiceProperty("SimpleCsvOutput", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(ISimpleTableOutput));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
