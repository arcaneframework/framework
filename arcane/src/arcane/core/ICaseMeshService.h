// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseMeshService.h                                          (C) 2000-2024 */
/*                                                                           */
/* Interface du service gérant un maillage du jeu de données.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEMESHSERVICE_H
#define ARCANE_CORE_ICASEMESHSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du service gérant les maillages du jeu de données.
 *
 * La création et l'initialisation se fait en 3 phases:
 * - une première phase lors de l'appel à createMesh() où le maillage est
 *   créé.. Dans cette phase, seule la classe gérant les maillage est créée
 *   mais ces derniers ne sont pas encore utilisables.
 * - une deuxième phase lors de l'appel à allocateMeshItems() où le maillage
 *   est alloué et initialisé. Cela correspond soit à la lecture du maillage,
 *   soit à la création dynamique des entités.
 * - une troisième phase qui consiste à partitionner le maillage si le code
 *   s'exécute en parallèle.
 * - une quatrième phase qui permet d'effectuer un traitement sur le maillage
 *   créé comme par exemple une sub-division.
 */
class ICaseMeshService
{
 public:

  virtual ~ICaseMeshService() = default;

 public:

  //! Créé le maillage avec le nom \a name
  virtual void createMesh(const String& name) = 0;

  //! Alloue les éléments du maillage
  virtual void allocateMeshItems() = 0;

  //! Partitionne le maillage.
  virtual void partitionMesh() = 0;

  //! Applique les opérations après tout le reste.
  virtual void applyAdditionalOperations() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
