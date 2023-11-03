﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshGlobal.h                                       (C) 2000-2023 */
/*                                                                           */
/* Déclarations de la composante 'arcane_cartesianmesh'.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHGLOBAL_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_cartesianmesh
#define ARCANE_CARTESIANMESH_EXPORT ARCANE_EXPORT
#else
#define ARCANE_CARTESIANMESH_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class CartesianMeshImpl;
class CellDirectionMng;
class NodeDirectionMng;
class FaceDirectionMng;
class ICartesianMesh;
class ICartesianMeshPatch;
class CartesianMeshPatch;
class CartesianConnectivity;
class CartesianMeshCoarsening;
class CartesianMeshCoarsening2;
class CartesianMeshRenumberingInfo;
class ICartesianMeshInternal;
class CartesianMeshPatchListView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Position des noeuds de la maille par direction pour les maillages
 * cartésiens.
 */
enum eCellNodePosition
{
  CNP_NextLeft = 0,
  CNP_NextRight = 1,
  CNP_PreviousRight = 2,
  CNP_PreviousLeft = 3,

  CNP_TopNextLeft = 4,
  CNP_TopNextRight = 5,
  CNP_TopPreviousRight = 6,
  CNP_TopPreviousLeft = 7
};

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
