// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSynchronizer.h                                  (C) 2000-2025 */
/*                                                                           */
/* Synchronisation de la liste des matériaux/milieux des entités.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_MESHMATERIALSYNCHRONIZER_H
#define ARCANE_MATERIALS_INTERNAL_MESHMATERIALSYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/core/VariableTypedef.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/materials/MatItem.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/materials/internal/IMeshMaterialSynchronizerImpl.h"
#include "arcane/materials/internal/AcceleratorMeshMaterialSynchronizerImpl.h"
#include "arcane/materials/internal/LegacyMeshMaterialSynchronizerImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
class MeshMaterialModifierImpl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Synchronisation de la liste des matériaux/milieux des entités.
 *
 * Cette classe permet de syncrhoniser entre les sous-domaines la liste
 * des matériaux/milieux auxquelles une maille appartient.
 *
 * Les mailles fantômes de ce sous-domaine vont récupérer des mailles propres
 * leur liste des matériaux/milieux. Ces mailles fantômes vont ensuite éventuellement
 * être ajoutés ou retirer des matériaux et milieux actuels pour être en cohérence
 * avec cette liste issue des mailles propres.
 */
class MeshMaterialSynchronizer
: public TraceAccessor
{
 public:

  explicit MeshMaterialSynchronizer(IMeshMaterialMng* material_mng);
  ~MeshMaterialSynchronizer();

 public:

  /*!
   * \brief Synchronisation de la liste des matériaux/milieux des entités.
   *
   * Cette opération est collective.
   *
   * Retourne \a true si des mailles ont été ajoutées ou supprimées d'un matériau
   * ou d'un milieu lors de cette opération pour ce sous-domaine.
   */
  bool synchronizeMaterialsInCells();
  void checkMaterialsInCells(Integer max_print);

 private:

  IMeshMaterialSynchronizerImpl* m_synchronizer;
  IMeshMaterialMng* m_material_mng;

  void _checkComponents(VariableCellInt32& indexes,
                        ConstArrayView<IMeshComponent*> components,
                        Integer max_print);
  void _checkComponentsInGhostCells(VariableCellInt64& hashes,
                                    Integer max_print);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
