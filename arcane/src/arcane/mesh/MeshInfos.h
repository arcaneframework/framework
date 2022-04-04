// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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

  //! Remet à zéro la numérotation
  void reset()
  {
    m_mesh_nb_node = 0;
    m_mesh_nb_edge = 0;
    m_mesh_nb_face = 0;
    m_mesh_nb_cell = 0;
  }

 private:

  Int32 m_mesh_rank;           //!< Numéro de ce sous-domaine
  Integer m_mesh_nb_node;      //!< Nombre de noeuds dans le maillage
  Integer m_mesh_nb_edge;      //!< Nombre d'arêtes dans le maillage
  Integer m_mesh_nb_face;      //!< Nombre de faces dans le maillage
  Integer m_mesh_nb_cell;      //!< Nombre de mailles dans le maillage
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* MESHINFOS_H_ */
