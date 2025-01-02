// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshGlobal.h                                                (C) 2000-2024 */
/*                                                                           */
/* Déclarations générales de la composante Maillage de Arcane.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHGLOBAL_H
#define ARCANE_MESH_MESHGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_MESH_BEGIN_NAMESPACE  namespace mesh {
#define ARCANE_MESH_END_NAMESPACE    }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemFamily;
class NodeFamily;
class EdgeFamily;
class FaceFamily;
class CellFamily;
class DynamicMesh;

class IncrementalItemConnectivity;

class ItemConnectivitySelector;
template<typename LeagcyType,typename CustomType>
class ItemConnectivitySelectorT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
