﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PolyhedralMesh.h                                            (C) 2000-2024 */
/*                                                                           */
/* Polyhedral mesh impl using Neo data structure                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_POLYHEDRALMESH_H
#define ARCANE_POLYHEDRALMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshHandle.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/MeshItemInternalList.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/Properties.h"
#include "arcane/core/ArcaneTypes.h"

#include "arcane/mesh/EmptyMesh.h"
#include "arcane/mesh/MeshEventsImpl.h"

#include "arcane/core/ItemAllocationInfo.h"
#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/String.h"
#include "arcane/utils/List.h"
#include "arcane/core/IMeshInitialAllocator.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshKind.h"

#include <memory>

#ifdef ARCANE_HAS_POLYHEDRAL_MESH_TOOLS
#include <vector>
#include <array>

#include "arcane/core/IVariableMng.h"
#include "DynamicMeshChecker.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISubDomain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

class PolyhedralMeshImpl;
class PolyhedralFamily;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PolyhedralMesh
: public EmptyMesh
, public IPolyhedralMeshInitialAllocator
{
 private:

  String m_name;
  ISubDomain* m_subdomain;
  MeshItemInternalList m_mesh_item_internal_list;
  inline static const String m_mesh_handle_name = "polyhedral_mesh_handle";
  MeshHandle m_mesh_handle;
  std::unique_ptr<Properties> m_properties;
  std::unique_ptr<PolyhedralMeshImpl> m_mesh; // using pimpl to limit dependency to neo lib to cc file
  IParallelMng* m_parallel_mng;
  MeshPartInfo m_mesh_part_info;
  bool m_is_allocated = false;
  ItemTypeMng* m_item_type_mng = nullptr;
  MeshKind m_mesh_kind;
  MeshEventsImpl m_mesh_events;

 public:

  // IPolyhedralMeshInitialAllocator interface
  void allocateItems(const Arcane::ItemAllocationInfo& item_allocation_info) override;

 public:

  PolyhedralMesh(ISubDomain* subDomain, const MeshBuildInfo& mbi);
  ~PolyhedralMesh(); // for pimpl idiom

 public:

  static String handleName() { return m_mesh_handle_name; }

  void read(String const& filename);

  void endUpdate();

  const MeshKind meshKind() const override { return m_mesh_kind; }

  EventObservable<const MeshEventArgs&>& eventObservable(eMeshEventType type) override
  {
    return m_mesh_events.eventObservable(type);
  }

#ifdef ARCANE_HAS_POLYHEDRAL_MESH_TOOLS

  class InitialAllocator : public IMeshInitialAllocator
  {
    PolyhedralMesh& m_mesh;

   public:

    explicit InitialAllocator(PolyhedralMesh& mesh)
    : m_mesh(mesh)
    {}
    IPolyhedralMeshInitialAllocator* polyhedralMeshAllocator() override { return &m_mesh; }
  };

 private:

  std::vector<std::unique_ptr<PolyhedralFamily>> m_arcane_families;
  std::array<std::unique_ptr<PolyhedralFamily>, NB_ITEM_KIND> m_empty_arcane_families;
  std::array<PolyhedralFamily*, NB_ITEM_KIND> m_default_arcane_families;
  std::vector<std::unique_ptr<VariableItemReal3>> m_arcane_item_coords;
  std::unique_ptr<VariableNodeReal3> m_arcane_node_coords = nullptr;
  ItemGroupList m_all_groups;
  InitialAllocator m_initial_allocator;
  IVariableMng* m_variable_mng;
  DynamicMeshChecker m_mesh_checker;
  List<IItemFamily*> m_item_family_collection;

  // IPrimaryMeshBase interface
  IMeshInitialAllocator* initialAllocator() override { return &m_initial_allocator; }

  // IMeshBase interface
 public:

  MeshHandle handle() const override;

 public:

  void build() override {}

  String name() const override;

  Integer nbNode() override;

  Integer nbEdge() override;

  Integer nbFace() override;

  Integer nbCell() override;

  Integer nbItem(eItemKind ik) override;

  ITraceMng* traceMng() override;

  Integer dimension() override;

  NodeGroup allNodes() override;

  EdgeGroup allEdges() override;

  FaceGroup allFaces() override;

  CellGroup allCells() override;

  NodeGroup ownNodes() override;

  EdgeGroup ownEdges() override;

  FaceGroup ownFaces() override;

  CellGroup ownCells() override;

  FaceGroup outerFaces() override;

  IItemFamily* createItemFamily(eItemKind ik, const String& name) override;

  ISubDomain* subDomain() override { return m_subdomain; }
  MeshItemInternalList* meshItemInternalList() override { return &m_mesh_item_internal_list; }

  Properties* properties() override { return m_properties.get(); }

  const MeshPartInfo& meshPartInfo() const override { return m_mesh_part_info; };

  IItemFamily* nodeFamily() override;
  IItemFamily* edgeFamily() override;
  IItemFamily* faceFamily() override;
  IItemFamily* cellFamily() override;

  InternalConnectivityPolicy _connectivityPolicy() const override { return InternalConnectivityPolicy::NewOnly; }

  IParallelMng* parallelMng() override { return m_subdomain->parallelMng(); }

  bool isAllocated() override { return m_is_allocated; }

  bool isAmrActivated() const override { return false; }

  IItemFamily* itemFamily(eItemKind ik) override;

  ItemTypeMng* itemTypeMng() const override;

  IItemFamily* findItemFamily(eItemKind ik, const String& name, bool create_if_needed, bool register_modifier_if_created) override;

  IMesh* parentMesh() const override { return nullptr; }

  PolyhedralFamily* arcaneDefaultFamily(eItemKind ik);

  VariableNodeReal3& nodesCoordinates() override;

  ItemGroup findGroup(const String& name) override;

  ItemGroupCollection groups() override;

  void destroyGroups() override;

  IGhostLayerMng* ghostLayerMng() const override { return nullptr; }

  void checkValidMesh() override
  {
    if (!m_is_allocated)
      return;
    m_mesh_checker.checkValidMesh();
  }

  IVariableMng* variableMng() const override { return m_variable_mng; }

  IItemFamilyCollection itemFamilies() override;

  String factoryName() const override;

 private:

  PolyhedralFamily* _createItemFamily(eItemKind ik, const String& name);
  PolyhedralFamily* _itemFamily(eItemKind ik);
  PolyhedralFamily* _findItemFamily(eItemKind ik, const String& name, bool create_if_needed = false);

#endif // ARCANE_HAS_POLYHEDRAL_MESH_TOOLS

 private:

  [[noreturn]] void _errorEmptyMesh() const;

  void _createUnitMesh();
  void _updateMeshInternalList(eItemKind kind);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_POLYHEDRALMESH_H
