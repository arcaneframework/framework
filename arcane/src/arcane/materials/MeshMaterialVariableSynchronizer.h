// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizer.h                          (C) 2000-2022 */
/*                                                                           */
/* Synchroniseur de variables matériaux.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLESYNCHRONIZER_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLESYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

#include "arcane/materials/IMeshMaterialVariableSynchronizer.h"
#include "arcane/materials/MatVarIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

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
  Ref<IMeshMaterialSynchronizeBuffer> commonBuffer() override { return m_commun_buffer; }

 private:

  IMeshMaterialMng* m_material_mng;
  IVariableSynchronizer* m_variable_synchronizer;
  UniqueArray< UniqueArray<MatVarIndex> > m_shared_items;
  UniqueArray< UniqueArray<MatVarIndex> > m_ghost_items;
  Int64 m_timestamp;
  MatVarSpace m_var_space;
  Ref<IMeshMaterialSynchronizeBuffer> m_commun_buffer;

 private:

  void _fillCells(Array<MatVarIndex>& items,AllEnvCellVectorView view);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

