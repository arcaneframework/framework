﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IInitialPartitioner.h                                       (C) 2000-2009 */
/*                                                                           */
/* Interface d'un partitionneur initial.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IINITIALPARTITIONER_H
#define ARCANE_IINITIALPARTITIONER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un partitionneur initial.
 *
 * Le service implémentant cette interface est responsable du
 * partitionnement initial des maillages du cas. Ce partitionnement a lieu
 * uniquement lors du démarrage du cas, juste avant l'initialisation
 * du cas.
 */
class IInitialPartitioner
{
 public:

  virtual ~IInitialPartitioner() {} //!< Libère les ressources.

 public:

  virtual void build() =0;

 public:

  /*!
   * \brief Partitionne les maillages.
   *
   * Cette opération doit partitionner tous les mailles \a meshes et
   * les distribuer sur l'ensemble des processeurs.
   */
  virtual void partitionAndDistributeMeshes(ConstArrayView<IMesh*> meshes) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
