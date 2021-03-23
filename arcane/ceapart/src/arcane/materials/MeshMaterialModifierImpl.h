// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialModifierImpl.h                                  (C) 2000-2015 */
/*                                                                           */
/* Implémentation de la modification des matériaux et milieux.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALMODIFIERIMPL_H
#define ARCANE_MATERIALS_MESHMATERIALMODIFIERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/IMeshMaterialModifierImpl.h"
#include "arcane/materials/IMeshMaterial.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialMng;
class IMeshMaterialVariable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialModifierImpl
: public TraceAccessor
, public IMeshMaterialModifierImpl
{
 private:
  
  class Operation;
  class OperationList
  {
   public:
    OperationList() : m_nb_direct_set(0){}
    ~OperationList();
   public:
    void add(Operation* o);
    void clear();
    ConstArrayView<Operation*> values() const { return m_operations.constView(); }
    Integer nbDirectSet() const { return m_nb_direct_set; }
   public:
    UniqueArray<Operation*> m_operations;
    Integer m_nb_direct_set;
  };

 public:

  MeshMaterialModifierImpl(MeshMaterialMng* mm);

 public:

  void initOptimizationFlags();

 public:

  virtual void addCells(IMeshMaterial* mat,Int32ConstArrayView ids);
  virtual void removeCells(IMeshMaterial* mat,Int32ConstArrayView ids);
  virtual void setCells(IMeshMaterial* mat,Int32ConstArrayView ids);

  virtual void endUpdate();
  virtual void beginUpdate();
  virtual void dumpStats();

 private:

  void _addCells(IMeshMaterial* mat,Int32ConstArrayView ids);
  void _setCells(IMeshMaterial* mat,Int32ConstArrayView ids);
  void _removeCells(IMeshMaterial* mat,Int32ConstArrayView ids);

  void _applyOperations();
  void _updateEnvironments();
  bool _checkMayOptimize();

 private:

  MeshMaterialMng* m_material_mng;
  OperationList m_operations;
  Integer nb_update;
  Integer nb_save_restore;
  Integer nb_optimize_add;
  Integer nb_optimize_remove;

  bool m_allow_optimization;
  bool m_allow_optimize_multiple_operation;
  bool m_allow_optimize_multiple_material;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

