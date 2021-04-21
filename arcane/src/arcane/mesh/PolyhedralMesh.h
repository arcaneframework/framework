// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PolyhedralMesh.h                                (C) 2000-2021             */
/*                                                                           */
/* Polyhedral mesh impl using Neo data structure                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANEFRAMEWORK_POLYHEDRALMESH_H
#define ARCANEFRAMEWORK_POLYHEDRALMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/ISubDomain.h"

#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS
#include "neo/Mesh.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT PolyhedralMesh{
 public :
  ISubDomain* m_subdomain;

 public:
  void read(String const& filename) {
    m_subdomain->traceMng()->info() << "--PolyhedralMesh : reading " << filename;
  }

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANEFRAMEWORK_POLYHEDRALMESH_H
