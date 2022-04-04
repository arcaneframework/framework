// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSynchronizer.h                                  (C) 2000-2016 */
/*                                                                           */
/* Synchronisation des entités des matériaux.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALSYNCHRONIZER_H
#define ARCANE_MATERIALS_MESHMATERIALSYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/MatItem.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableMng;
class Properties;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialModifierImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation d'un gestion des matériaux.
 */
class MeshMaterialSynchronizer
: public TraceAccessor
{
 public:

  MeshMaterialSynchronizer(IMeshMaterialMng* material_mng);
  ~MeshMaterialSynchronizer();

 public:

  void synchronizeMaterialsInCells();
  void checkMaterialsInCells(Integer max_print);

 private:

  IMeshMaterialMng* m_material_mng;

  inline static void _setBit(ByteArrayView bytes,Integer position);
  inline static bool _hasBit(ByteConstArrayView bytes,Integer position);
  void _fillPresence(AllEnvCell all_env_cell,ByteArrayView presence);
  void _checkComponents(VariableCellInt32& indexes,
                        ConstArrayView<IMeshComponent*> components,
                        Integer max_print);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
