// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PolyhedralMesh.cc                                     (C) 2000-2021       */
/*                                                                           */
/* Polyhedral mesh impl using Neo data structure                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/PolyhedralMesh.h"

#include "arcane/ISubDomain.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/mesh/ItemFamily.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
_errorEmptyMesh() const {
  m_subdomain->traceMng()->fatal() << "Cannot use PolyhedralMesh if Arcane is not linked with lib Neo";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane {
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace mesh {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PolyhedralFamily : public ItemFamily{
  IMesh* m_mesh;
 public:
  PolyhedralFamily(IMesh* mesh, eItemKind ik, String name)
  : m_mesh(mesh)
  , ItemFamily(mesh,ik,name){}

 public:
  void addItems(Int64ConstArrayView uids, Int32ArrayView items) {
    m_mesh->traceMng()->info() << " PolyhedralFamily::ADDITEMS " ;
  }
};

} // namespace mesh
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS

#include "arcane/IMeshMng.h"
#include "arcane/MeshHandle.h"
#include "arcane/IItemFamily.h"
#include "arcane/mesh/ItemFamily.h"

#include "neo/Mesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

  class PolyhedralMeshImpl
  {
    ISubDomain* m_subdomain;
    Neo::Mesh m_mesh{ "Test" };

    template <eItemKind IK>
    class ItemKindTraits {static const Neo::ItemKind item_kind = Neo::ItemKind::IK_None;};

    static Neo::ItemKind itemKindArcaneToNeo(eItemKind ik)
    {
      switch (ik) {
      case (IK_Cell):
        return Neo::ItemKind::IK_Cell;
      case (IK_Face):
        return Neo::ItemKind::IK_Face;
      case (IK_Edge):
        return Neo::ItemKind::IK_Edge;
      case (IK_Node):
        return Neo::ItemKind::IK_Node;
      case (IK_DoF):
        return Neo::ItemKind::IK_Dof;
      case (IK_Unknown):
      case (IK_Particle):
        return Neo::ItemKind::IK_None;
      }
      return Neo::ItemKind::IK_Node;
    }

   public:
    class ItemLocalIds {
      Neo::FutureItemRange m_future_items;
     public :
      friend class PolyhedralMeshImpl;
    };

   public:
    PolyhedralMeshImpl(ISubDomain* subDomain)
    : m_subdomain(subDomain)
    {}

   public:
    void read(const String& filename)
    {
      m_subdomain->traceMng()->info() << "--PolyhedralMesh : reading " << filename;
      // First step create a single cell
      _createSingleCellTest();
    }

    String name() const { return m_mesh.name(); }

    Integer dimension() const { return m_mesh.dimension(); }

    Integer nbNode() const { return m_mesh.nbNodes(); }
    Integer nbEdge() const { return m_mesh.nbEdges(); }
    Integer nbFace() const { return m_mesh.nbFaces(); }
    Integer nbCell() const { return m_mesh.nbCells(); }
    Integer nbItem(eItemKind ik) const { return m_mesh.nbItems(itemKindArcaneToNeo(ik)); }
    ItemGroup allCells() { return ItemGroup{}; }

    void addFamily(eItemKind ik, const String& name){
      m_mesh.addFamily(itemKindArcaneToNeo(ik),name.localstr());
    }

    void scheduleAddItems(PolyhedralFamily* arcane_item_family,
                          Int64ConstArrayView uids,
                          ItemLocalIds& item_local_ids) noexcept(ndebug) {
      auto& added_items = item_local_ids.m_future_items;
      auto& item_family = m_mesh.findFamily(itemKindArcaneToNeo(arcane_item_family->itemKind()),
                                            arcane_item_family->name().localstr());
      m_mesh.scheduleAddItems(item_family, std::vector<Int64>{uids.begin(), uids.end()}, added_items);
      // add arcane items
      auto & mesh_graph = m_mesh.internalMeshGraph();
      String arcane_item_lids_property_name {"Arcane_Item_Lids"};
      item_family.addProperty<Neo::utils::Int32>(arcane_item_lids_property_name.localstr());
      mesh_graph.addAlgorithm(Neo::InProperty{item_family,item_family.lidPropName()},
                              Neo::OutProperty{item_family,arcane_item_lids_property_name.localstr()},
                              [arcane_item_family,uids]
                              (Neo::ItemLidsProperty const& lids_property,
                               Neo::PropertyT<Neo::utils::Int32> & arcane_item_lids){
                                Int32UniqueArray arcane_items(uids.size());
                                arcane_item_family->addItems(uids,arcane_items);
                              });
    }

    void applyScheduledOperations() noexcept {
      m_mesh.applyScheduledOperations();
    }

   private:
    void _createSingleCellTest()
    {
      auto& cell_family = m_mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
      auto& node_family = m_mesh.addFamily(Neo::ItemKind::IK_Node, "node_family");
      auto added_cells = Neo::FutureItemRange{};
      m_mesh.scheduleAddItems(cell_family, { 0 }, added_cells);
      auto added_nodes = Neo::FutureItemRange{};
      m_mesh.scheduleAddItems(node_family, { 0, 1, 2, 3, 4, 5 }, added_nodes);
      m_mesh.applyScheduledOperations();
    }
  };

  template <> class PolyhedralMeshImpl::ItemKindTraits<IK_Cell> {static const Neo::ItemKind item_kind = Neo::ItemKind::IK_Cell;};
  template <> class PolyhedralMeshImpl::ItemKindTraits<IK_Face> {static const Neo::ItemKind item_kind = Neo::ItemKind::IK_Face;};
  template <> class PolyhedralMeshImpl::ItemKindTraits<IK_Edge> {static const Neo::ItemKind item_kind = Neo::ItemKind::IK_Edge;};
  template <> class PolyhedralMeshImpl::ItemKindTraits<IK_Node> {static const Neo::ItemKind item_kind = Neo::ItemKind::IK_Node;};
  template <> class PolyhedralMeshImpl::ItemKindTraits<IK_DoF> {static const Neo::ItemKind item_kind = Neo::ItemKind::IK_Dof;};

} // End namespace mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

mesh::PolyhedralMesh::
~PolyhedralMesh()
{
  m_mesh_handle._setMesh(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* mesh::PolyhedralMesh::
traceMng()
{
  return m_subdomain->traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const MeshHandle& mesh::PolyhedralMesh::handle() const
{
  return m_mesh_handle;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mesh::PolyhedralMesh::
_errorEmptyMesh() const
{
  ARCANE_FATAL("Cannot use PolyhedralMesh if Arcane is not linked with lib Neo");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS
// All PolyhedralMesh methods must be defined twice (emtpy in the second case)

mesh::PolyhedralMesh::
PolyhedralMesh(ISubDomain* subdomain)
: EmptyMesh{subdomain->traceMng()}
, m_subdomain{subdomain}
, m_mesh_handle{m_subdomain->meshMng()->createMeshHandle(m_mesh_handle_name)}
, m_properties(std::make_unique<Properties>(subdomain->propertyMng(),String("ArcaneMeshProperties_")+m_name))
, m_mesh{ std::make_unique<mesh::PolyhedralMeshImpl>(m_subdomain) }
{
  m_mesh_handle._setMesh(this);
  m_default_arcane_families.fill(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
read(const String& filename)
{
  // First step: create manually a unit mesh
  ARCANE_UNUSED(filename); // temporary
  _createUnitMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Arcane::mesh::PolyhedralMesh::
name() const
{
  return m_mesh->name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
dimension()
{
  return m_mesh->dimension();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbNode()
{
  return m_mesh->nbNode();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbEdge()
{
  return m_mesh->nbEdge();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbFace()
{
  return m_mesh->nbFace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbCell()
{
  return m_mesh->nbCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbItem(eItemKind ik)
{
  return m_mesh->nbItem(ik);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup mesh::PolyhedralMesh::
allCells()
{
  return m_mesh->allCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
createItemFamily(eItemKind ik, const String& name)
{
  m_mesh->addFamily(ik, name);
  m_arcane_families.push_back(std::make_unique<PolyhedralFamily>(this,ik, name));
  auto current_family = m_arcane_families.back().get();
  if (m_default_arcane_families[ik] == nullptr) m_default_arcane_families[ik] = current_family;
  return current_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void mesh::PolyhedralMesh::
_createUnitMesh()
{
  createItemFamily(IK_Cell, "CellFamily");
  createItemFamily(IK_Node, "NodeFamily");
  auto cell_family = m_default_arcane_families[IK_Cell];
  auto node_family = m_default_arcane_families[IK_Node];
  Int64UniqueArray cell_uids{0},node_uids{ 0, 1, 2, 3, 4, 5 };
  // todo add a cell_lids struct (containing future)
  PolyhedralMeshImpl::ItemLocalIds cell_lids,node_lids;
  m_mesh->scheduleAddItems(cell_family, cell_uids, cell_lids);
  m_mesh->scheduleAddItems(node_family, node_uids, node_lids);
  m_mesh->applyScheduledOperations();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
nodeFamily()
{
  return m_default_arcane_families[IK_Node];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
edgeFamily()
{
  return m_default_arcane_families[IK_Edge];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
faceFamily()
{
  return m_default_arcane_families[IK_Face];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* mesh::PolyhedralMesh::
cellFamily()
{
  return m_default_arcane_families[IK_Cell];
}

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#else // ARCANE_HAS_CUSTOM_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class PolyhedralMeshImpl{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::mesh::PolyhedralMesh::
~PolyhedralMesh() = default;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Arcane::mesh::PolyhedralMesh::
PolyhedralMesh(ISubDomain* subdomain)
: m_subdomain{subdomain}
, m_mesh{nullptr}
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


void Arcane::mesh::PolyhedralMesh::
read([[maybe_unused]] const String& filename)
{
  _errorEmptyMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
dimension()
{
  _errorEmptyMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbNode()
{
  _errorEmptyMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbEdge()
{
  _errorEmptyMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbFace()
{
  _errorEmptyMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbCell()
{
  _errorEmptyMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer Arcane::mesh::PolyhedralMesh::
nbItem([[maybe_unused]] eItemKind ik)
{
  _errorEmptyMesh();
}


#endif // ARCANE_HAS_CUSTOM_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/