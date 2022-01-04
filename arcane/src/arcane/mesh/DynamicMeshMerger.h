// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshMerger.h                                         (C) 2000-2018 */
/*                                                                           */
/* Fusion de plusieurs maillages.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_DYNAMICMESHMERGER_H
#define ARCANE_MESH_DYNAMICMESHMERGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/mesh/MeshGlobal.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace mesh
{
class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour un échange de maillage entre sous-domaines.
 */
class ARCANE_MESH_EXPORT DynamicMeshMerger
: public TraceAccessor
{
 public:

  DynamicMeshMerger(DynamicMesh* mesh);
  ~DynamicMeshMerger();

 public:

  void mergeMeshes(ConstArrayView<DynamicMesh*> meshes);

 private:

  DynamicMesh* m_mesh;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mesh
} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
