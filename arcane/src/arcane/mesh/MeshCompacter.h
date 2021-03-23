// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCompacter.h                                             (C) 2000-2016 */
/*                                                                           */
/* Gestion d'un compactage de familles du maillage.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHCOMPACTER_H
#define ARCANE_MESH_MESHCOMPACTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/List.h"
#include "arcane/IMeshCompacter.h"
#include "arcane/mesh/MeshGlobal.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
class ITimeStats;
class IItemFamilyCompactPolicy;
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion d'un compactage de familles du maillage..
 *
 * Les instances de cette classe sont créée par l'appel à
 * MeshCompactMng::beginCompact().
 */
class ARCANE_MESH_EXPORT MeshCompacter
: public TraceAccessor
, public IMeshCompacter
{
  friend class MeshCompactMng;

 private:

  typedef std::map<IItemFamily*,ItemFamilyCompactInfos*> ItemFamilyCompactInfosMap;

 private:

  MeshCompacter(IMesh* mesh,ITimeStats* stats);
  MeshCompacter(IItemFamily* family,ITimeStats* stats);
  ~MeshCompacter();

 private:

  void build();

 public:

  IMesh* mesh() const override;
  void doAllActions() override;
  void beginCompact() override;
  void compactVariablesAndGroups() override;
  void updateInternalReferences() override;
  void endCompact() override;
  void finalizeCompact() override;
  const ItemFamilyCompactInfos* findCompactInfos(IItemFamily* family) const override;
  ePhase phase() const override { return m_phase; }
  bool isSorted() const override { return m_is_sorted; }
  ItemFamilyCollection families() const override;

 public:

  void setSorted(bool v) override;
  void _setCompactVariablesAndGroups(bool v) override;

 private:

  IMesh* m_mesh;
  ItemFamilyCompactInfosMap m_family_compact_infos_map;
  ITimeStats* m_time_stats;
  ePhase m_phase;
  bool m_is_sorted;
  bool m_is_compact_variables_and_groups;
  List<IItemFamily*> m_item_families;

 private:

  void _checkPhase(ePhase wanted_phase);
  void _addFamily(IItemFamily* family);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
