﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseMeshMasterService.h                                    (C) 2000-2020 */
/*                                                                           */
/* Interface du service gérant les maillages du jeu de données.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ICASEMESHMASTERSERVICE_H
#define ARCANE_ICASEMESHMASTERSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface du service gérant les maillages du jeu de données.
 *
 * Ce service gère les différents services de création/lecture de maillage.
 *
 * Cela se fait en deux phases:
 * - une première phase lors de l'appel à createMeshes() où l'ensemble des
 *   maillages issus du jeu de données est créée. Dans cette phase, seule
 *   la classe gérant les maillage est créé mais ces derniers ne sont pas
 *   encore utilisables.
 * - une deuxième phase lors de l'appel à allocateMeshes() où les maillages
 *   sont effectivement alloués. Cela correspond soit à la lecture des maillage,
 *   soit à la création effective des entités qu'ils gèrent.
 *
 * Il existe une troisième phase optionnelle qui n'est effectuée que en
 * parallèle et qui consiste à partitionner les maillages, via l'appel
 * à partitionMeshes().
 */
class ICaseMeshMasterService
{
 public:
  virtual ~ICaseMeshMasterService() = default;
 public:
  //! Créé les maillages
  virtual void createMeshes() =0;
  //! Créé les maillages
  virtual void allocateMeshes() =0;
  //! Partitionne les maillages
  virtual void partitionMeshes() =0;
 public:
  virtual ICaseOptions* _options() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
