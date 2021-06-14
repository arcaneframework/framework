// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
#include "arcane/utils/Collection.h"
#include "arcane/MeshHandle.h"
#include "arcane/ItemGroup.h"
#include "arcane/mesh/EmptyMesh.h"
#include "arcane/MeshItemInternalList.h"
#include "arcane/ISubDomain.h"
#include "arcane/Properties.h"
#include "arcane/ArcaneTypes.h"

#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS
#include <vector>
#include <array>
#include <memory>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane {
class ISubDomain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


namespace Arcane::mesh {

class PolyhedralMeshImpl;
class PolyhedralFamily;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PolyhedralMesh : public EmptyMesh {
 public :
  String m_name = "polyhedral_mesh";
  ISubDomain* m_subdomain;
  MeshItemInternalList m_mesh_item_internal_list;
  inline static const String m_mesh_handle_name = "polyhedral_mesh_handle";
  MeshHandle m_mesh_handle;
  std::unique_ptr<Properties> m_properties;
  std::unique_ptr<PolyhedralMeshImpl> m_mesh; // using pimpl to limit dependency to neo lib to cc file
  MeshPartInfo m_part_info;

 public:
  PolyhedralMesh(ISubDomain* subDomain);
  ~PolyhedralMesh(); // for pimpl idiom

 public:
  static String handleName() { return m_mesh_handle_name; }

  void read(String const& filename);

#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS

 private:
  std::vector<std::unique_ptr<PolyhedralFamily>> m_arcane_families;
  std::array<PolyhedralFamily*,NB_ITEM_KIND> m_default_arcane_families;

  // IMeshBase interface
 public:

  const MeshHandle& handle() const override;

 public:

  String name() const override;

  Integer nbNode() override;

  Integer nbEdge() override;

  Integer nbFace() override;

  Integer nbCell() override;

  Integer nbItem(eItemKind ik) override;

  ITraceMng* traceMng() override;

  Integer dimension() override;

  NodeGroup allNodes() override { return ItemGroup{}; }

  EdgeGroup allEdges() override { return ItemGroup{}; }

  FaceGroup allFaces() override { return ItemGroup{}; }

  CellGroup allCells() override;

  NodeGroup ownNodes() override { return ItemGroup{}; }

  EdgeGroup ownEdges() override { return ItemGroup{}; }

  FaceGroup ownFaces() override { return ItemGroup{}; }

  CellGroup ownCells() override { return ItemGroup{}; }

  FaceGroup outerFaces() override { return ItemGroup{}; }

  IItemFamily* createItemFamily(eItemKind ik,const String& name) override;

  ISubDomain* subDomain() override { return m_subdomain; }
  MeshItemInternalList* meshItemInternalList() override { return &m_mesh_item_internal_list; }

  Properties* properties() override { return m_properties.get(); }

  const MeshPartInfo& meshPartInfo() const { return m_part_info; };

  IItemFamily* nodeFamily();
  IItemFamily* edgeFamily();
  IItemFamily* faceFamily();
  IItemFamily* cellFamily();

#endif // ARCANE_HAS_CUSTOM_MESH_TOOLS

 private:
  [[noreturn]] void _errorEmptyMesh() const;

  void _createUnitMesh();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_POLYHEDRALMESH_H
