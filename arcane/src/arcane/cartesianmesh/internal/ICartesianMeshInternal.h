﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshInternal.h                                    (C) 2000-2023 */
/*                                                                           */
/* Partie interne à Arcane de ICartesianMesh.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_INTERNAL_ICARTESIANMESHINTERNAL_H
#define ARCANE_CARTESIANMESH_INTERNAL_ICARTESIANMESHINTERNAL_H
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
 * \brief Partie interne de ICartesianMesh.
 */
class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshInternal
{
 public:

  virtual ~ICartesianMeshInternal() = default;

 public:

  /*!
   * \brief Créé une instance pour gérer le déraffinement du maillage (V2).
   * \warning Experimental method !
   */
  virtual Ref<CartesianMeshCoarsening2> createCartesianMeshCoarsening2() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
