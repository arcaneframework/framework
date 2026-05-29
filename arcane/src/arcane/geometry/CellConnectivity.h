// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellConnectivity.h                                          (C) 2000-2026 */
/*                                                                           */
/* Information on the connectivity of a mesh.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_CELLCONNECTIVITY_H
#define ARCANE_GEOMETRIC_CELLCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometry/GeomType.h"
#include "arcane/geometry/GeometricConnectic.h"
#include "arcane/geometry/ItemStaticInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::geometric
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Information on the connectivity of a mesh.
 */
class CellConnectivity
{
 public:

  CellConnectivity(Integer nb_node, Integer nb_edge, Integer nb_face, GeomType cell_type)
  : m_nb_node(nb_node)
  , m_nb_edge(nb_edge)
  , m_nb_face(nb_face)
  , m_cell_type(cell_type)
  {
  }

 public:

  Integer nbNode() const { return m_nb_node; }
  Integer nbEdge() const { return m_nb_edge; }
  Integer nbFace() const { return m_nb_face; }

  //! Number of control sub-volumes
  Integer nbSubZone() const { return m_nb_sub_zone; }
  //! Number of internal SVC faces
  Integer nbSubZoneFace() const { return m_nb_svc_face; }
  const Integer* nodeAssociation() const { return m_node_association; }
  const NodeConnectic* nodeConnectic() const { return m_node_connectic; }
  const EdgeConnectic* edgeConnectic() const { return m_edge_connectic; }
  const FaceConnectic* faceConnectic() const { return m_face_connectic; }
  const SVCFaceConnectic* SCVFaceConnectic() const { return m_svc_face_connectic; }
  //! Mesh type (GeomType::Quad4, GeomType::Hexaedron8, ...)
  GeomType cellType() const { return m_cell_type; }

 protected:

  Integer m_nb_sub_zone = 0; //!< Number of control sub-volumes
  Integer m_nb_svc_face = 0; //!< Number of internal SVC faces
  //! Local number of the vertex associated with the control sub-volume
  const Integer* m_node_association = nullptr;
  const NodeConnectic* m_node_connectic = nullptr;
  const EdgeConnectic* m_edge_connectic = nullptr;
  const FaceConnectic* m_face_connectic = nullptr;
  const SVCFaceConnectic* m_svc_face_connectic = nullptr;

 public:

  //! Local numbers in the control sub-volumes
  Integer m_edge_node_sub_zone_id[3];
  Integer m_face_node_sub_zone_id[3];
  //! Connectic for edges
  Integer m_edge_direct_connectic[ItemStaticInfo::MAX_CELL_EDGE * 2];

 protected:

  Integer m_nb_node;
  Integer m_nb_edge;
  Integer m_nb_face;
  Int32* m_edge_first_node = nullptr;
  Int32* m_edge_second_node = nullptr;
  GeomType m_cell_type;

 protected:

  inline void _setEdgeDirectConnectic();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Connectivity information for geometric elements of type IT_NullType.
 */
class NullConnectivity
: public CellConnectivity
{
 public:

  NullConnectivity()
  : CellConnectivity(0, 0, 0, GeomType::NullType)
  {
    _init();
  }

 public:

  Integer nbNode() const { return 0; }
  Integer nbEdge() const { return 0; }
  Integer nbFace() const { return 0; }

 public:
 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometry/GeneratedConnectivity.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::geometric

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
