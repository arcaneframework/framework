// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshBase.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Interface for base mesh operations                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHBASE_H
#define ARCANE_CORE_IMESHBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for base mesh operations.
 *
 * This interface is created to gradually implement IMesh operations in a
 * new implementation.
 *
 * This interface should be temporary.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemFamilyModifier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMeshBase
{
 public:

  virtual ~IMeshBase() = default;

 public:

  //! Handle sur ce maillage
  virtual MeshHandle handle() const = 0;

 public:

  //! Nom du maillage
  virtual String name() const = 0;

  //! Nombre de noeuds du maillage
  virtual Integer nbNode() = 0;

  //! Nombre d'arêtes du maillage
  virtual Integer nbEdge() = 0;

  //! Nombre de faces du maillage
  virtual Integer nbFace() = 0;

  //! Nombre de mailles du maillage
  virtual Integer nbCell() = 0;

  /*!
   * \brief Nombre d'éléments du genre \a ik.
   * \pre ik==IK_Node || ik==IK_Edge || ik==IK_Face || ik==IK_Cell
   */
  virtual Integer nbItem(eItemKind ik) = 0;

  //! Gestionnaire de message associé
  virtual ITraceMng* traceMng() = 0;

  /*!
   * \brief Dimension du maillage (1D, 2D ou 3D).
   *
   * La dimension correspond à la dimension des éléments mailles (Cell).
   * Si des mailles de plusieurs dimensions sont présentes, c'est la dimension
   * la plus importante qui est retournée.
   * Si la dimension n'a pas encore été positionnée, retourne -1;
   */
  virtual Integer dimension() = 0;

  //! Groupe de tous les noeuds
  virtual NodeGroup allNodes() = 0;

  //! Groupe de toutes les arêtes
  virtual EdgeGroup allEdges() = 0;

  //! Groupe de toutes les faces
  virtual FaceGroup allFaces() = 0;

  //! Groupe de toutes les mailles
  virtual CellGroup allCells() = 0;

  //! Groupe de tous les noeuds propres au domaine
  virtual NodeGroup ownNodes() = 0;

  //! Groupe de toutes les arêtes propres au domaine
  virtual EdgeGroup ownEdges() = 0;

  //! Groupe de toutes les faces propres au domaine
  virtual FaceGroup ownFaces() = 0;

  //! Groupe de toutes les mailles propres au domaine
  virtual CellGroup ownCells() = 0;

  //! Groupe de toutes les faces sur la frontière.
  virtual FaceGroup outerFaces() = 0;

 public:

  //! Créé une famille de particule de nom \a name
  virtual IItemFamily* createItemFamily(eItemKind ik, const String& name) = 0;

  /*!
   * \brief Retourne la famille de nom \a name.
   *
   * Si \a create_if_needed est vrai, la famille est créé si elle n'existait pas.
   * Si \a register_modifier_if_created est vrai, le modifier de la famille est enregistré
   */
  virtual IItemFamily* findItemFamily(eItemKind ik, const String& name, bool create_if_needed = false,
                                      bool register_modifier_if_created = false) = 0;

  /*!
   * \brief Retourne la famille de nom \a name.
   *
   * Si la famille demandée n'existe pas, si \a throw_exception vaut \a true une
   * exception est levée, sinon le pointeur nul est retourné.
   */
  virtual IItemFamily* findItemFamily(const String& name, bool throw_exception = false) = 0;

  /*!
   * \brief Retourne l'interface IItemFamilyModifier pour famille de nom \a name et de type \a ik
   *
   * Si ce modificateur n'est pas trouvé, retourne nullptr
   */
  virtual IItemFamilyModifier* findItemFamilyModifier(eItemKind ik, const String& name) = 0;

  /*!
   * \brief Retourne la famille d'entité de type \a ik.
   *
   * \pre ik==IK_Node || ik==IK_Edge || ik==IK_Face || ik==IK_Cell
   */
  virtual IItemFamily* itemFamily(eItemKind ik) = 0;

  //! Retourne la famille des noeuds.
  virtual IItemFamily* nodeFamily() = 0;
  //! Retourne la famille des arêtes.
  virtual IItemFamily* edgeFamily() = 0;
  //! Retourne la famille des faces.
  virtual IItemFamily* faceFamily() = 0;
  //! Retourne la famille des mailles.
  virtual IItemFamily* cellFamily() = 0;

  virtual IItemFamilyCollection itemFamilies() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_IMESHBASE_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
