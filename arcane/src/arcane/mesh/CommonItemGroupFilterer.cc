// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonItemGroupFilterer.cc                                  (C) 2000-2021 */
/*                                                                           */
/* Filtrage des groupes communs à toutes les parties d'un maillage.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/CommonItemGroupFilterer.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/SerializeBuffer.h"
#include "arcane/IParallelMng.h"
#include "arcane/IItemFamily.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommonItemGroupFilterer::
CommonItemGroupFilterer(IItemFamily* family)
: m_family(family)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CommonItemGroupFilterer::
addGroupToFilter(const ItemGroup& group)
{
  m_input_groups.add(group);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CommonItemGroupFilterer::
applyFiltering()
{
  UniqueArray<ItemGroup> groups_to_check;
  for( List<ItemGroup>::Enumerator i(m_input_groups); ++i; )
    groups_to_check.add(*i);
  IParallelMng* pm = m_family->parallelMng();
  ITraceMng* tm = m_family->traceMng();

  Integer nb_group = groups_to_check.size();
  tm->info(4) << "CHECK: nb_group_to_compare=" << nb_group;

  // Créé un buffer pour sérialiser les noms des groupes
  SerializeBuffer send_buf;
  send_buf.setMode(ISerializer::ModeReserve);
  send_buf.reserveInteger(1);
  for( Integer i=0; i< nb_group; ++i ){
    send_buf.reserve(groups_to_check[i].fullName());
  }

  send_buf.allocateBuffer();
  send_buf.setMode(ISerializer::ModePut);
  send_buf.putInteger(nb_group);
  for( Integer i=0; i< nb_group; ++i ){
    send_buf.put(groups_to_check[i].fullName());
  }

  // Récupère les infos des autres PE.
  SerializeBuffer recv_buf;
  pm->allGather(&send_buf,&recv_buf);

  std::map<String,Int32> group_occurences;

  Int32 nb_rank = pm->commSize();
  recv_buf.setMode(ISerializer::ModeGet);
  for( Integer i=0; i<nb_rank; ++i ){
    Integer nb_group_rank = recv_buf.getInteger();
    for( Integer z=0; z< nb_group_rank; ++z ){
      String x;
      recv_buf.get(x);
      auto vo = group_occurences.find(x);
      if (vo== group_occurences.end())
        group_occurences.insert(std::make_pair(x,1));
      else
        vo->second = vo->second + 1;
    }
  }

  // Parcours la liste des groupes et range dans \a common_groups
  // ceux qui sont disponibles sur tous les rangs de \a pm.
  // Cette liste sera triée par ordre alphabétique.
  std::map<String,ItemGroup> common_groups;
  UniqueArray<String> bad_groups;
  {
    auto end_var = group_occurences.end();
    for( Integer i=0; i< nb_group; ++i ){
      ItemGroup group = groups_to_check[i];
      String group_name = group.fullName();
      auto i_group = group_occurences.find(group_name);
      if (i_group ==end_var)
        // Ne devrait pas arriver
        continue;
      if (i_group->second!=nb_rank){
        bad_groups.add(group_name);
        continue;
      }
      common_groups.insert(std::make_pair(group_name, group));
    }
  }

  if (!bad_groups.empty())
    ARCANE_FATAL("The following ItemGroup are not on all mesh parts: {0}",bad_groups);

  m_sorted_common_groups.clear();
  for( const auto& x : common_groups )
    m_sorted_common_groups.add(x.second);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

