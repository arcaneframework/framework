// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellConnectivity.h                                          (C) 2000-2014 */
/*                                                                           */
/* Informations sur la connectivité d'une maille.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_CELLCONNECTIVITY_H
#define ARCANE_GEOMETRIC_CELLCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometric/GeomType.h"
#include "arcane/geometric/GeometricConnectic.h"
#include "arcane/geometric/ItemStaticInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Informations sur la connectivité d'une maille.
 */
class CellConnectivity
{
 public:
  CellConnectivity(Integer nb_node,Integer nb_edge,Integer nb_face,GeomType cell_type)
  : m_nb_node(nb_node), m_nb_edge(nb_edge), m_nb_face(nb_face), m_cell_type(cell_type)
  {
    m_nb_sub_zone = 0;
    m_nb_svc_face = 0;
    m_node_association = 0;
    m_node_connectic = 0;
    m_edge_connectic = 0;
    m_face_connectic = 0;
    m_svc_face_connectic = 0;
  }

 public:

  Integer nbNode() const { return m_nb_node; }
  Integer nbEdge() const { return m_nb_edge; }
  Integer nbFace() const { return m_nb_face; }

  //! Nombre de sous volume de controle
  Integer nbSubZone() const { return m_nb_sub_zone; }
  //! Nombre de sous faces internes SVC
  Integer nbSubZoneFace() const { return m_nb_svc_face; }
  const Integer* nodeAssociation() const { return m_node_association; }
  const NodeConnectic* nodeConnectic() const { return m_node_connectic; }
  const EdgeConnectic* edgeConnectic() const { return m_edge_connectic; }
  const FaceConnectic* faceConnectic() const { return m_face_connectic; }
  const SVCFaceConnectic* SCVFaceConnectic() const { return m_svc_face_connectic; }
  //! Type de la maille (GeomType::Quad4, GeomType::Hexaedron8, ...)
  GeomType cellType() const { return m_cell_type; }

 protected:

  Integer m_nb_sub_zone; //!< Nombre de sous volume de controle
  Integer m_nb_svc_face; //!< Nombre de sous faces internes SVC
  //! Numero local du sommet associé au sous volume de controle
  const Integer *m_node_association;
  const NodeConnectic *m_node_connectic;
  const EdgeConnectic *m_edge_connectic;
  const FaceConnectic *m_face_connectic;
  const SVCFaceConnectic *m_svc_face_connectic;

 public:

  //! Numéro locaux dans le sous-volumes de contrôle
  Integer m_edge_node_sub_zone_id[3];
  Integer m_face_node_sub_zone_id[3];
  //! Connectique pour les arêtes
  Integer m_edge_direct_connectic[ItemStaticInfo::MAX_CELL_EDGE*2];

 protected:

  Integer m_nb_node;
  Integer m_nb_edge;
  Integer m_nb_face;
  Int32* m_edge_first_node;
  Int32* m_edge_second_node;
  GeomType m_cell_type;

 protected:

  inline void _setEdgeDirectConnectic();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Informations de connectivité pour les éléments géométriques de type IT_NullType.
 */
class NullConnectivity
: public CellConnectivity
{
 public:
  NullConnectivity()
  : CellConnectivity(0,0,0,GeomType::NullType)
  { _init(); }
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

#include "arcane/geometric/GeneratedConnectivity.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
