// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCompacter.cc                                            (C) 2000-2022 */
/*                                                                           */
/* Management of a mesh exchange between subdomains.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Timer.h"

#include "arcane/ItemFamilyCompactInfos.h"
#include "arcane/IItemFamily.h"
#include "arcane/IItemFamilyPolicyMng.h"
#include "arcane/IItemFamilyCompactPolicy.h"

#include "arcane/mesh/MeshCompacter.h"
#include "arcane/mesh/DynamicMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCompacter::
MeshCompacter(IMesh* mesh,ITimeStats* stats)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_time_stats(stats)
, m_phase(ePhase::Init)
, m_is_sorted(false)
, m_is_compact_variables_and_groups(true)
{
  for( IItemFamily* family : m_mesh->itemFamilies() ){
    _addFamily(family);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCompacter::
MeshCompacter(IItemFamily* family,ITimeStats* stats)
: TraceAccessor(family->traceMng())
, m_mesh(family->mesh())
, m_time_stats(stats)
, m_phase(ePhase::Init)
, m_is_sorted(false)
, m_is_compact_variables_and_groups(true)
{
  _addFamily(family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCompacter::
~MeshCompacter()
{
  for( const auto& i : m_family_compact_infos_map )
    delete i.second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
_addFamily(IItemFamily* family)
{
  // Only adds the family if it has a compaction policy.
  IItemFamilyCompactPolicy* c = family->policyMng()->compactPolicy();
  if (c)
    m_item_families.add(family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
build()
{
  m_phase = ePhase::BeginCompact;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
_checkPhase(ePhase wanted_phase)
{
  if (m_phase!=wanted_phase)
    ARCANE_FATAL("Invalid exchange phase wanted={0} current={1}",
                 (int)wanted_phase,(int)m_phase);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
beginCompact()
{
  _checkPhase(ePhase::BeginCompact);

  // Starts a compaction and calculates the correspondences between the
  // new and old localIds().
  {
    Timer::Action ts_action(m_time_stats,"CompactItemsBegin");
    for( IItemFamily* family : m_item_families ){
      auto compact_infos = new ItemFamilyCompactInfos(this,family);
      m_family_compact_infos_map.insert(std::make_pair(family,compact_infos));
    }
    for( IItemFamily* family : m_item_families ){
      IItemFamilyCompactPolicy* c = family->policyMng()->compactPolicy();
      auto iter = m_family_compact_infos_map.find(family);
      if (iter==m_family_compact_infos_map.end())
        ARCANE_FATAL("Can not find family '{0}'",family->name());
      c->beginCompact(*iter->second);
    }
  }

  m_phase = ePhase::CompactVariableAndGroups;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
compactVariablesAndGroups()
{
  _checkPhase(ePhase::CompactVariableAndGroups);

  // Updates groups and variables once the parents are updated.
  // This must only be done on the families that are compacted.
  if (m_is_compact_variables_and_groups){
    Timer::Action ts_action(m_time_stats,"CompactVariables");
    for( const auto& iter : m_family_compact_infos_map ){
      const ItemFamilyCompactInfos* compact_infos = iter.second;
      IItemFamilyCompactPolicy* c = compact_infos->family()->policyMng()->compactPolicy();
      c->compactVariablesAndGroups(*compact_infos);
    }
  }

  m_phase = ePhase::UpdateInternalReferences;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
updateInternalReferences()
{
  _checkPhase(ePhase::UpdateInternalReferences);

  // Updates the references of each entity.
  {
    Timer::Action ts_action(m_time_stats,"CompactUpdateInternalReferences");
    for( IItemFamily* family : m_mesh->itemFamilies() ){
      IItemFamilyCompactPolicy* c = family->policyMng()->compactPolicy();
      if (c)
        c->updateInternalReferences(this);
    }
  }

  m_phase = ePhase::EndCompact;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
endCompact()
{
  _checkPhase(ePhase::EndCompact);

  // Ends the compaction.
  {
    Timer::Action ts_action(m_time_stats,"CompactItemFinish");

    for( const auto& iter : m_family_compact_infos_map ){
      ItemFamilyCompactInfos* compact_infos = iter.second;
      IItemFamilyCompactPolicy* c = compact_infos->family()->policyMng()->compactPolicy();
      c->endCompact(*compact_infos);
    }
  }

  m_phase = ePhase::Finalize;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
finalizeCompact()
{
  _checkPhase(ePhase::Finalize);

  // Notifies that the compaction is finished.
  {
    Timer::Action ts_action(m_time_stats,"CompactItemFinish");

    for( IItemFamily* family : m_mesh->itemFamilies() ){
      IItemFamilyCompactPolicy* c = family->policyMng()->compactPolicy();
      if (c)
        c->finalizeCompact(this);
    }
  }

  m_phase = ePhase::Ended;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ItemFamilyCompactInfos* MeshCompacter::
findCompactInfos(IItemFamily* family) const
{
  auto x = m_family_compact_infos_map.find(family);
  if (x==m_family_compact_infos_map.end())
    return nullptr;
  return x->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMesh* MeshCompacter::
mesh() const
{
  return m_mesh;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
doAllActions()
{
  beginCompact();
  compactVariablesAndGroups();
  updateInternalReferences();
  endCompact();
  finalizeCompact();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
setSorted(bool v)
{
  _checkPhase(ePhase::BeginCompact);
  m_is_sorted = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCompacter::
_setCompactVariablesAndGroups(bool v)
{
  _checkPhase(ePhase::BeginCompact);
  m_is_compact_variables_and_groups = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemFamilyCollection MeshCompacter::
families() const
{
  return m_item_families;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
