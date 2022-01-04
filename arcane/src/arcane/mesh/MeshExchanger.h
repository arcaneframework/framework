// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshExchanger.h                                             (C) 2000-2018 */
/*                                                                           */
/* Gestion d'un échange de maillage entre sous-domaines.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHEXCHANGER_H
#define ARCANE_MESHEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/List.h"
#include "arcane/IMeshExchanger.h"
#include "arcane/mesh/MeshGlobal.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
class ITimeStats;
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour un échange de maillage entre sous-domaines.
 */
class ARCANE_MESH_EXPORT MeshExchanger
: public TraceAccessor
, public IMeshExchanger
{
 private:

  typedef std::map<IItemFamily*,IItemFamilyExchanger*> ItemFamilyExchangerMap;

 public:

  MeshExchanger(DynamicMesh* mesh,ITimeStats* stats);
  ~MeshExchanger();

 public:

  bool computeExchangeInfos() override;
  void processExchange() override;
  void removeNeededItems() override;
  void allocateReceivedItems() override;
  void updateItemGroups() override;
  void updateVariables() override;
  void finalizeExchange() override;
  IPrimaryMesh* mesh() const override;
  void build();
  IItemFamilyExchanger* findExchanger(IItemFamily* family) override;
  ePhase phase() const override { return m_phase; }

 protected:

  void _setNextPhase(ePhase next_phase);

 private:

  DynamicMesh* m_mesh;
  List<IItemFamilyExchanger*> m_family_exchangers;
  ItemFamilyExchangerMap m_family_exchanger_map;
  ITimeStats* m_time_stats;
  ePhase m_phase;

  void _checkPhase(ePhase wanted_phase);
  void _buildWithItemFamilyNetwork();
  void _addItemFamilyExchanger(IItemFamily* family);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
