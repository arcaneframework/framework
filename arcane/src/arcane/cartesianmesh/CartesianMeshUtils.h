// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshUtils.h                                        (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires associées à 'ICartesianMesh'.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHUTILS_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ref.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions utilitaires associées à 'ICartesianMesh'.
 */
namespace Arcane::CartesianMeshUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Créé une instance pour gérer le déraffinement du maillage (V2).
 * \warning Experimental method !
 */
extern "C++" ARCANE_CARTESIANMESH_EXPORT Ref<CartesianMeshCoarsening2>
createCartesianMeshCoarsening2(ICartesianMesh* cm);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::CartesianMeshUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
