// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizer.h                          (C) 2000-2017 */
/*                                                                           */
/* Synchroniseur de variables matériaux.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLESYNCHRONIZER_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLESYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/IMeshMaterialVariableSynchronizer.h"
#include "arcane/materials/MatVarIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AllEnvCellVectorView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Synchroniseur de variables matériaux.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariableSynchronizer
: public TraceAccessor
, public IMeshMaterialVariableSynchronizer
{
 public:

  MeshMaterialVariableSynchronizer(IMeshMaterialMng* material_mng,
                                   IVariableSynchronizer* var_syncer,
                                   MatVarSpace mvs);

  virtual ~MeshMaterialVariableSynchronizer() override;

 public:

  IVariableSynchronizer* variableSynchronizer() override;
  ConstArrayView<MatVarIndex> sharedItems(Int32 index) override;
  ConstArrayView<MatVarIndex> ghostItems(Int32 index) override;
  void recompute() override;
  void checkRecompute() override;

 private:

  IMeshMaterialMng* m_material_mng;
  IVariableSynchronizer* m_variable_synchronizer;
  UniqueArray< SharedArray<MatVarIndex> > m_shared_items;
  UniqueArray< SharedArray<MatVarIndex> > m_ghost_items;
  Int64 m_timestamp;
  MatVarSpace m_var_space;

 private:

  void _fillCells(Array<MatVarIndex>& items,AllEnvCellVectorView view);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

