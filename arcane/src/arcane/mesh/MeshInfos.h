// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshInfos.h                                                 (C) 2000-2017 */
/*                                                                           */
/* General mesh information                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHINFOS_H_ 
#define ARCANE_MESHINFOS_H_ 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Classe factorisant des informations sur le maillage
 *
 * Attention cette classe ne donne pas d'information sur la maillage construit (utiliser IMesh).
 * Il s'agit simplement d'un conteneur utilisé pendant la phase de construction du maillage
 */

class ARCANE_MESH_EXPORT MeshInfos
{
public:

  /** Constructeur de la classe */
  MeshInfos(const Integer rank)
  : m_mesh_rank(rank)
  , m_mesh_nb_node(0)
  , m_mesh_nb_edge(0)
  , m_mesh_nb_face(0)
  , m_mesh_nb_cell(0){}

  /** Destructeur de la classe */
  virtual ~MeshInfos() {}

public:
  //! Numéro de ce sous-domaine
  Int32 rank() const {return m_mesh_rank;}

  //! Nombre de noeuds dans le maillage
  Integer& nbNode() {return m_mesh_nb_node;}
  Integer getNbNode() const {return m_mesh_nb_node;}

  //! Nombre d'arêtes dans le maillage
  Integer& nbEdge() {return m_mesh_nb_edge;}
  Integer getNbEdge() const {return m_mesh_nb_edge;}

  //! Nombre de faces dans le maillage
  Integer& nbFace() {return m_mesh_nb_face;}
  Integer getNbFace() const {return m_mesh_nb_face;}


  //! Nombre de mailles dans le maillage
  Integer& nbCell() {return m_mesh_nb_cell;}
  Integer getNbCell() const {return m_mesh_nb_cell;}

private:

  Int32 m_mesh_rank;           //!< Numéro de ce sous-domaine
  Integer m_mesh_nb_node;      //!< Nombre de noeuds dans le maillage
  Integer m_mesh_nb_edge;      //!< Nombre d'arêtes dans le maillage
  Integer m_mesh_nb_face;      //!< Nombre de faces dans le maillage
  Integer m_mesh_nb_cell;      //!< Nombre de mailles dans le maillage
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* MESHINFOS_H_ */
