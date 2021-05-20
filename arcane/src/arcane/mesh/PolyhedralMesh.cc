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
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
_errorEmptyMesh() const {
  m_subdomain->traceMng()->fatal() << "Cannot use PolyhedralMesh if Arcane is not linked with lib Neo";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS

#include "arcane/IMeshMng.h"
#include "arcane/MeshHandle.h"
#include "arcane/IItemFamily.h"
#include "arcane/mesh/ItemFamily.h"

#include "neo/Mesh.h"

#include <vector>
#include <memory>

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
: m_subdomain{subdomain}
, m_mesh_handle{m_subdomain->meshMng()->createMeshHandle(m_mesh_handle_name)}
, m_mesh{ std::make_unique<mesh::PolyhedralMeshImpl>(m_subdomain) }
{
  m_mesh_handle._setMesh(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
read(const String& filename)
{
  m_mesh->read(filename);
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


#else // ARCANE_HAS_CUSTOM_MESH_TOOLS : empty class for compilation

namespace mesh
{
  class PolyhedralMeshImpl{};
}

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


String Arcane::mesh::PolyhedralMesh::
name() const
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


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

#endif // ARCANE_HAS_CUSTOM_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/