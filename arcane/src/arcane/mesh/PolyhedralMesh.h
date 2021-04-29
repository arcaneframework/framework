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
#ifndef ARCANE_POLYHEDRALMESH_H
#define ARCANE_POLYHEDRALMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <memory>
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/ISubDomain.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh {

class PolyhedralMeshImpl;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT PolyhedralMesh{
 public :
  ISubDomain* m_subdomain;

  std::unique_ptr<PolyhedralMeshImpl> m_mesh; // using pimpl to limit dependency to neo lib to cc file

 public:
  PolyhedralMesh(ISubDomain* subDomain);
  ~PolyhedralMesh(); // for pimpl idiom

 public:
  void read(String const& filename);

 private:
  void _errorEmptyMesh() {
    m_subdomain->traceMng()->fatal() << "Cannot use PolyhedralMesh if Arcane is not linked with lib Neo";
  }

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_POLYHEDRALMESH_H
