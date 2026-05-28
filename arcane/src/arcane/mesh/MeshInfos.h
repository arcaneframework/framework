// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshInfos.h                                                 (C) 2000-2022 */
/*                                                                           */
/* General mesh information                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHINFOS_H
#define ARCANE_MESH_MESHINFOS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Class containing mesh information
 *
 * Warning: This class does not provide information about the constructed mesh (use IMesh).
 * It is simply a container used during the mesh construction phase.
 */

class ARCANE_MESH_EXPORT MeshInfos
{
 public:

  /** Class constructor */
  MeshInfos(const Integer rank)
  : m_mesh_rank(rank)
  , m_mesh_nb_node(0)
  , m_mesh_nb_edge(0)
  , m_mesh_nb_face(0)
  , m_mesh_nb_cell(0)
  {}

  /** Class destructor */
  virtual ~MeshInfos() {}

 public:

  //! Number of this subdomain
  Int32 rank() const { return m_mesh_rank; }

  //! Number of nodes in the mesh
  Integer& nbNode() { return m_mesh_nb_node; }
  Integer getNbNode() const { return m_mesh_nb_node; }

  //! Number of edges in the mesh
  Integer& nbEdge() { return m_mesh_nb_edge; }
  Integer getNbEdge() const { return m_mesh_nb_edge; }

  //! Number of faces in the mesh
  Integer& nbFace() { return m_mesh_nb_face; }
  Integer getNbFace() const { return m_mesh_nb_face; }

  //! Number of cells in the mesh
  Integer& nbCell() { return m_mesh_nb_cell; }
  Integer getNbCell() const { return m_mesh_nb_cell; }

  //! Resets the numbering
  void reset()
  {
    m_mesh_nb_node = 0;
    m_mesh_nb_edge = 0;
    m_mesh_nb_face = 0;
    m_mesh_nb_cell = 0;
  }

 private:

  Int32 m_mesh_rank; //!< Number of this subdomain
  Integer m_mesh_nb_node; //!< Number of nodes in the mesh
  Integer m_mesh_nb_edge; //!< Number of edges in the mesh
  Integer m_mesh_nb_face; //!< Number of faces in the mesh
  Integer m_mesh_nb_cell; //!< Number of cells in the mesh
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
