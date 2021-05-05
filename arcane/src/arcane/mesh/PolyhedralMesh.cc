// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PolyhedralMesh.cc                                     (C) 2000-2021       */
/*                                                                           */
/* Polyhedral mesh impl using Neo data structure                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMeshMng.h"
#include "arcane/MeshHandle.h"
#include "arcane/ISubDomain.h"
#include "arcane/utils/ITraceMng.h"

#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS
#include "neo/Mesh.h"
#endif

#include "arcane/mesh/PolyhedralMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS

namespace mesh
{

  class PolyhedralMeshImpl
  {
    ISubDomain* m_subdomain;
    Neo::Mesh m_mesh{ "Test" };

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

   private:
    void _createSingleCellTest()
    {
      auto cell_family = m_mesh.addFamily(Neo::ItemKind::IK_Cell, "cell_family");
      auto node_family = m_mesh.addFamily(Neo::ItemKind::IK_Node, "node_family");
      auto added_cells = Neo::FutureItemRange{};
      m_mesh.scheduleAddItems(cell_family, { 0 }, added_cells);
      auto added_nodes = Neo::FutureItemRange{};
      m_mesh.scheduleAddItems(node_family, { 0, 1, 2, 3, 4, 5 }, added_nodes);
      m_mesh.applyScheduledOperations();
    }
  };

} // End namespace mesh
#endif // End ARCANE_HAS_CUSTOM_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

mesh::PolyhedralMesh::~PolyhedralMesh() = default;

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
_errorEmptyMesh() const {
  m_subdomain->traceMng()->fatal() << "Cannot use PolyhedralMesh if Arcane is not linked with lib Neo";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS
// All PolyhedralMesh methods must be defined twice (emtpy in the second case)

mesh::PolyhedralMesh::
PolyhedralMesh(ISubDomain* subdomain)
: m_subdomain{subdomain}
, m_mesh{ std::make_unique<mesh::PolyhedralMeshImpl>(m_subdomain) }
, m_mesh_handle{m_subdomain->meshMng()->createMeshHandle(m_mesh_handle_name)}
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
  return String{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_HAS_CUSTOM_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/







/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

