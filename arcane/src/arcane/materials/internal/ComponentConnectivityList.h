// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentConnectivityList.h                                 (C) 2000-2023 */
/*                                                                           */
/* Gestion des listes de connectivité des milieux et matériaux.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_COMPONENTCONNECTIVITYLIST_H
#define ARCANE_MATERIALS_INTERNAL_COMPONENTCONNECTIVITYLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/core/VariableTypes.h"

#include "arcane/materials/MaterialsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion des listes de connectivité des milieux et matériaux.
 */
class ComponentConnectivityList
: public TraceAccessor
{
  class ComponentContainer;
  class Container;

 public:

  explicit ComponentConnectivityList(MeshMaterialMng* mm);
  ~ComponentConnectivityList();

 public:

  ComponentConnectivityList(ComponentConnectivityList&&) = delete;
  ComponentConnectivityList(const ComponentConnectivityList&) = delete;
  ComponentConnectivityList& operator=(ComponentConnectivityList&&) = delete;
  ComponentConnectivityList& operator=(const ComponentConnectivityList&) = delete;

 public:

  void endCreate(bool is_continue);

  void addCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids);
  void removeCellsToEnvironment(Int16 env_id, ConstArrayView<Int32> cell_ids);

  void addCellsToMaterial(Int16 mat_id, ConstArrayView<Int32> cell_ids);
  void removeCellsToMaterial(Int16 mat_id, ConstArrayView<Int32> cell_ids);

  const VariableCellInt16& cellNbEnvironment() const;
  const VariableCellInt16& cellNbMaterial() const;

 private:

  MeshMaterialMng* m_material_mng = nullptr;
  Container* m_container = nullptr;

 private:

  void _addCells(Int16 env_id, ConstArrayView<Int32> cell_ids, ComponentContainer& component);
  void _removeCells(Int16 env_id, ConstArrayView<Int32> cell_ids, ComponentContainer& component);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
