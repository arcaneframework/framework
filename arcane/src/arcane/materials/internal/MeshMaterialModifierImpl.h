// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialModifierImpl.h                                  (C) 2000-2023 */
/*                                                                           */
/* Implémentation de la modification des matériaux et milieux.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIALMODIFIERIMPL_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIALMODIFIERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/internal/IMeshMaterialModifierImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class MeshMaterialMng;
class IMeshMaterialVariable;
class MaterialModifierOperation;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialModifierImpl
: public TraceAccessor
, public IMeshMaterialModifierImpl
{
 private:
  
  using Operation = MaterialModifierOperation;

  class OperationList
  {
   public:
    ~OperationList();
   public:
    void add(Operation* o);
    void clear();
    ConstArrayView<Operation*> values() const { return m_operations.constView(); }
   public:
    UniqueArray<Operation*> m_operations;
  };

 public:

  explicit MeshMaterialModifierImpl(MeshMaterialMng* mm);

 public:

  void initOptimizationFlags();

 public:

  void addCells(IMeshMaterial* mat,Int32ConstArrayView ids) override;
  void removeCells(IMeshMaterial* mat,Int32ConstArrayView ids) override;

  void endUpdate() override;
  void beginUpdate() override;
  void dumpStats() override;

 private:

  void _addCells(IMeshMaterial* mat,Int32ConstArrayView ids);
  void _removeCells(IMeshMaterial* mat,Int32ConstArrayView ids);

  void _applyOperations();
  void _updateEnvironments();
  bool _checkMayOptimize();

 private:

  MeshMaterialMng* m_material_mng = nullptr;
  OperationList m_operations;
  Integer nb_update = 0;
  Integer nb_save_restore = 0;
  Integer nb_optimize_add = 0;
  Integer nb_optimize_remove = 0;

  bool m_allow_optimization = false;
  bool m_allow_optimize_multiple_operation = false;
  bool m_allow_optimize_multiple_material = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

