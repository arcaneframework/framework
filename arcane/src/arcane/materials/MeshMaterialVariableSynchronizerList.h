﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizerList.h                      (C) 2000-2023 */
/*                                                                           */
/* Liste de variables à synchroniser.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLESYNCHRONIZERLIST_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLESYNCHRONIZERLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/MatVarIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Synchronisation d'une liste de variables matériaux.
 *
 * La méthode add() permet d'ajouter des variables à synchroniser.
 * Il faut ensuite appeler apply() pour effectuer la synchronisation.
 *
 * Une instance de ce cette classe peut-être utilisée plusieurs fois.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariableSynchronizerList
{
 private:

  class Impl;
  class SyncInfo;

 public:

  explicit MeshMaterialVariableSynchronizerList(IMeshMaterialMng* material_mng);
  ~MeshMaterialVariableSynchronizerList();

 public:

  MeshMaterialVariableSynchronizerList(const MeshMaterialVariableSynchronizerList&) = delete;
  MeshMaterialVariableSynchronizerList& operator=(const MeshMaterialVariableSynchronizerList&) = delete;
  MeshMaterialVariableSynchronizerList(const MeshMaterialVariableSynchronizerList&&) = delete;
  MeshMaterialVariableSynchronizerList& operator=(const MeshMaterialVariableSynchronizerList&&) = delete;

 public:

  //! Effectue la synchronisation
  void apply();

  //! Ajoute la variable \a var à la liste des variables à synchroniser
  void add(MeshMaterialVariable* var);

  //! Après appel à apply(), contient la taille des messages envoyés
  Int64 totalMessageSize() const;

  /*!
   * \brief Commence une synchronisation non bloquante.
   *
   * Cela est valide uniquement si IMeshMaterialMng::synchronizeVariableVersion() vaut 7.
   */
  void beginSynchronize();

  /*!
   * \brief Bloque tant que la synchronisation en cours n'est pas terminé.
   *
   * Il faut appeler beginSynchronize() avant cet appel.
   */
  void endSynchronize();

 private:

  Impl* m_p;

 private:

  static void _beginSynchronizeMultiple(SyncInfo& sync_info);
  static void _beginSynchronizeMultiple2(SyncInfo& sync_info);
  static void _endSynchronizeMultiple2(SyncInfo& sync_info);
  void _fillSyncInfo(SyncInfo& sync_info);
  void _beginSynchronize(bool is_blocking);
  void _endSynchronize(bool is_blocking);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

