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

#include "arcane/mesh/PolyhedralMesh.h"

#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS
#include "neo/Mesh.h"
#endif

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
    Neo::Mesh mesh{"Test"};

   public:
    PolyhedralMeshImpl(ISubDomain* subDomain) : m_subdomain(subDomain){}

   public:
    void read(const String& filename)
    {
      m_subdomain->traceMng()->info() << "--PolyhedralMesh : reading " << filename;
    }
  };

}

#endif // End ARCANE_HAS_CUSTOM_MESH_TOOLS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS
// All PolyhedralMesh methods must be defined twice (emtpy in the second case)

Arcane::mesh::PolyhedralMesh::
PolyhedralMesh(ISubDomain* subdomain)
: m_subdomain{subdomain}
, m_mesh{ std::make_unique<mesh::PolyhedralMeshImpl>(m_subdomain) }
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
read(const String& filename)
{
  m_mesh->read(filename);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#else // ARCANE_HAS_CUSTOM_MESH_TOOLS : empty class for compilation

Arcane::mesh::PolyhedralMesh::
PolyhedralMesh(ISubDomain* subdomain)
: m_subdomain{subdomain}
, m_mesh{nullptr}
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Arcane::mesh::PolyhedralMesh::
read(const String& filename)
{
  _errorEmptyMesh();
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

