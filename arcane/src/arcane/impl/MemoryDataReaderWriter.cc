// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryDataReaderWriter.cc                                   (C) 2000-2009 */
/*                                                                           */
/* Lecture/ecriture des données en mémoire.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Ref.h"

#include "arcane/IData.h"
#include "arcane/IVariable.h"
#include "arcane/VariableCollection.h"

#include "arcane/impl/MemoryDataReaderWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryDataReaderWriter::
~MemoryDataReaderWriter()
{
  free();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryDataReaderWriter::
free()
{
  m_vars_to_data.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryDataReaderWriter::
beginWrite(const VariableCollection& vars)
{
  // Copie la table courante et enlève toutes les références aux variables
  // que l'on va sauver. Une fois ceci terminé, il ne reste que des
  // références à des variables plus utilisées. On libère donc
  // le IData correspondant.
  VarToDataMap vars_to_data = m_vars_to_data;
  for( VariableCollection::Enumerator ivar(vars); ++ivar; ){
    IVariable* var = *ivar;
    VarToDataMap::iterator i = vars_to_data.find(var->fullName());
    if (i!=vars_to_data.end())
      vars_to_data.erase(i);
  }

  for( VarToDataMap::iterator i=vars_to_data.begin(); i!=vars_to_data.end(); ++i ){
    // Supprime la référence à la table courante et détruit la donnée
    m_vars_to_data.erase(i->first);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryDataReaderWriter::
write(IVariable* var,IData* data)
{
  Ref<IData> cdata = _findData(var);
  if (!cdata.get()){
    cdata = data->cloneRef();
    m_vars_to_data.insert(std::make_pair(var->fullName(),cdata));
  }
  else{
    cdata->copy(data);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryDataReaderWriter::
read(IVariable* var,IData* data)
{
  Ref<IData> cdata = _findData(var);
  if (!cdata.get()){
    warning() << A_FUNCNAME << ": "
              << String::format("can not find data for variable '{0}': variable will not be restored",
                                var->fullName());
    return;
  }
  data->copy(cdata.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IData> MemoryDataReaderWriter::
_findData(IVariable* var)
{
  auto i = m_vars_to_data.find(var->fullName());
  if (i==m_vars_to_data.end())
    return {};
  return i->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

