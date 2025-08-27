// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshAccessor.h                                              (C) 2000-2020 */
/*                                                                           */
/* Accès aux informations d'un maillage.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHACCESSOR_H
#define ARCANE_MESHACCESSOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;
class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Accès aux informations d'un maillage.
 * \ingroup Mesh
 */
class ARCANE_CORE_EXPORT MeshAccessor
{
 public:

  ARCCORE_DEPRECATED_2020("Use constructor with MeshHande")
  MeshAccessor(ISubDomain* sd);
  MeshAccessor(IMesh* mesh);
  MeshAccessor(const MeshHandle& mesh_handle);

 public:

  //! Retourne le nombre de mailles du maillage
  Integer nbCell() const;
  //! Retourne le nombre de faces du maillage
  Integer nbFace() const;
  //! Retourne le nombre d'arêtes du maillage
  Integer nbEdge() const;
  //! Retourne le nombre de noeuds du maillage
  Integer nbNode() const;

  //! Retourne les coordonnées des noeuds du maillage
  VariableNodeReal3& nodesCoordinates() const;

  //! Retourne le groupe contenant tous les noeuds
  NodeGroup allNodes() const;
  //! Retourne le groupe contenant toutes les arêtes
  EdgeGroup allEdges() const;
  //! Retourne le groupe contenant toutes les faces
  FaceGroup allFaces() const;
  //! Retourne le groupe contenant toutes les mailles
  CellGroup allCells() const;
  //! Retourne le groupe contenant toutes les faces de le frontière.
  FaceGroup outerFaces() const;
  /*! \brief Retourne le groupe contenant tous les noeuds propres à ce domaine.
   *
   * En mode séquentiel, il s'agit de allNodes(). En mode parallèle, il s'agit de
   * tous les noeuds qui ne sont pas des noeuds fantômes. L'ensemble des
   * groupes ownNodes() de tous les sous-domaines forment une partition
   * du maillage global.
   */
  NodeGroup ownNodes() const;
  /*! \brief Retourne le groupe contenant toutes les mailles propres à ce domaine.
   *
   * En mode séquentiel, il s'agit de allCells(). En mode parallèle, il s'agit de
   * toutes les mailles qui ne sont pas des mailles fantômes. L'ensemble des
   * groupes ownCells() de tous les sous-domaines forment une partition
   * du maillage global.
   */
  CellGroup ownCells() const;
  /*! \brief Groupe contenant toutes les faces propres à ce domaine.
   *
   * En mode séquentiel, il s'agit de allFaces(). En mode parallèle, il s'agit de
   * toutes les faces qui ne sont pas des faces fantômes. L'ensemble des
   * groupes ownFaces() de tous les sous-domaines forment une partition
   * du maillage global.
   */
  FaceGroup ownFaces() const;
  /*! \brief Groupe contenant toutes les arêtes propres à ce domaine.
   *
   * En mode séquentiel, il s'agit de allEdges(). En mode parallèle, il s'agit de
   * toutes les arêtes qui ne sont pas des arêtes fantômes. L'ensemble des
   * groupes ownEdges() de tous les sous-domaines forment une partition
   * du maillage global.
   */
  EdgeGroup ownEdges() const;


 public:

  inline IMesh* mesh() const { return m_mesh_handle.mesh(); }
  inline const MeshHandle& meshHandle() const { return m_mesh_handle; }

 private:

  MeshHandle m_mesh_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
