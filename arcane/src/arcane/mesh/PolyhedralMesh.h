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
#include "arcane/MeshHandle.h"
#include "arcane/IMeshBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane {
class ISubDomain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


namespace Arcane::mesh {

class PolyhedralMeshImpl;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT PolyhedralMesh : public IMeshBase {
 public :
  ISubDomain* m_subdomain;
  String m_mesh_handle_name;
  MeshHandle m_mesh_handle;

  std::unique_ptr<PolyhedralMeshImpl> m_mesh; // using pimpl to limit dependency to neo lib to cc file

 public:
  PolyhedralMesh(ISubDomain* subDomain);
  ~PolyhedralMesh(); // for pimpl idiom

 public:
  void read(String const& filename);

  // IMeshBase interface
 public:

  //! Handle sur ce maillage
  const MeshHandle& handle() const override;

 public:

  String name() const override;

  Integer nbNode() override { return -1; }

  Integer nbEdge() override { return -1; }

  Integer nbFace() override { return -1; }

  Integer nbCell() override { return -1; }

  Integer nbItem(eItemKind ik) override { return -1; }

  ITraceMng* traceMng() override;

  Integer dimension() override { return -1; }

 private:
  void _errorEmptyMesh();

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_POLYHEDRALMESH_H
