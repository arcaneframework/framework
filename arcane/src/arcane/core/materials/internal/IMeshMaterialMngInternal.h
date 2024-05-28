// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialMngInternal.h                                  (C) 2000-2024 */
/*                                                                           */
/* API interne Arcane de 'IMeshMaterialMng'.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALMNGINTERNAL_H
#define ARCANE_CORE_MATERIALS_INTERNAL_IMESHMATERIALMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"
#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief API interne Arcane de 'IMeshMaterialMng'.
 */
class ARCANE_CORE_EXPORT IMeshMaterialMngInternal
{
 public:

  virtual ~IMeshMaterialMngInternal() = default;

 public:

  /*!
   * \brief Ajoute la variable \a var.
   *
   * Cette méthode ne doit pas être appelée directement. Les références
   * aux variables l'appelle si nécessaire. Cette méthode doit être
   * appelée avec le verrou variableLock() actif.
   */
  virtual void addVariable(IMeshMaterialVariable* var) = 0;

  /*!
   * \brief Supprime la variable \a var.
   *
   * Cette méthode ne doit pas être appelée directement. Les références
   * aux variables l'appelle si nécessaire. Cette méthode doit être
   * appelée avec le verrou variableLock() actif. A noter que cette
   * fonction n'appelle pas l'opérateur delete sur \a var.
   */
  virtual void removeVariable(IMeshMaterialVariable* var) = 0;

  /*!
   * \brief Implémentation du modificateur.
   *
   * Ce modificateur permet de changer la liste des mailles composant un milieu
   * ou un matériau. Cette méthode ne doit en principe pas être appelée directement.
   * Pour modifier, il vaut mieux utiliser une instance de MeshMaterialModifier
   * qui garantit que les fonctions de mise à jour sont bien appelées.
   */
  virtual MeshMaterialModifierImpl* modifier() = 0;

  /*!
   * \brief Liste des infos pour indexer les variables matériaux.
   */
  virtual ConstArrayView<MeshMaterialVariableIndexer*> variablesIndexer() = 0;

  /*!
   * \brief Synchronizeur pour les variables matériaux et milieux sur toutes les mailles.
   */
  virtual IMeshMaterialVariableSynchronizer* allCellsMatEnvSynchronizer() = 0;

  /*!
   * \brief Synchronizeur pour les variables uniquement milieux sur toutes les mailles.
   */
  virtual IMeshMaterialVariableSynchronizer* allCellsEnvOnlySynchronizer() = 0;

 public:

  /*!
   * \brief Renvoie la table de "connectivité" CellLocalId -> AllEnvCell
   * destinée à être utilisée dans un RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL
   * en conjonction de la macro ENUMERATE_CELL_ALLENVCELL
   */
  virtual AllCellToAllEnvCell* getAllCellToAllEnvCell() const = 0;

  /*!
   * \brief Construit la table de "connectivité" CellLocalId -> AllEnvCell
   * destinée à être utilisée dans un RUNCOMMAND_ENUMERATE_CELL_ALLENVCELL
   * en conjonction de la macro ENUMERATE_CELL_ALLENVCELL
   *
   * Si aucun allocateur n'est spécifié alors la méthode
   * platform::getDefaultDataAllocator() est utilisée
   */
  virtual void createAllCellToAllEnvCell(IMemoryAllocator* alloc) = 0;

  /*!
   * \briefInstance de ComponentItemSharedInfo pour un constituant
   *
   * La valeur de \a level doit être LEVEL_MATERIAL ou LEVEL_ENVIRONMENT
   */
  virtual ComponentItemSharedInfo* componentItemSharedInfo(Int32 level) const = 0;

  //! File d'exécution associée
  virtual RunQueue& runQueue() const = 0;

  //! Liste de files asynchrones
  virtual Accelerator::RunQueuePool& asyncRunQueuePool() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
