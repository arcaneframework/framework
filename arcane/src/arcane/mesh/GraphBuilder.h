// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GraphBuilder.h                                              (C) 2000-2013 */
/*                                                                           */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_SRC_ARCANE_MESH_GRAPHBUILDER_H_
#define ARCANE_SRC_ARCANE_MESH_GRAPHBUILDER_H_

#include "arcane/IGraph2.h"
#include "arcane/mesh/GraphDoFs.h"
#include "arcane/mesh/ParticleFamily.h"
#include "arcane/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GraphBuilder {
public :
 static IGraph2* createGraph(IMesh* mesh, String const& particle_family_name=ParticleFamily::defaultFamilyName())
 {
   return new mesh::GraphDoFs(mesh,particle_family_name);
 };

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif /* ARCANE_SRC_ARCANE_MESH_GRAPHBUILDER_H_ */

