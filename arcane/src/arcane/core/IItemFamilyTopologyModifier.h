// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyTopologyModifier.h                               (C) 2000-2025 */
/*                                                                           */
/* Interface de modification de la topologie des entités d'une famille.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMFAMILYTOPOLOGYMODIFIER_H
#define ARCANE_IITEMFAMILYTOPOLOGYMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Interface de modification de la topologie des entités d'une famille.
 *
 * Cette classe à vocation à être temporaire et sert pour remplacer
 * les appels direct à ItemInternal en gérant les anciennes ou nouvelles
 * connectivités.
 *
 * \warning Cette classe permet de modifier directement les connectivités
 * des maillages non-structurés ce qui peut introduire des incohérences dans
 * la topologie et les connectivités. Il faut donc faire très attention
 * lorsqu'on utilise des méthodes de cette classe. Il est préférable
 * d'utiliser les méthodes de IMeshModifier si on souhaite ajouter/supprimer
 * des entités du maillage en garantissant la cohérence.
 */
class ARCANE_CORE_EXPORT IItemFamilyTopologyModifier
{
 public:

  //! Libère les ressources
  virtual ~IItemFamilyTopologyModifier() = default;

 public:

  //! Famille associée
  virtual IItemFamily* family() const = 0;

 public:

  /*!
   * \brief Remplace un noeud d'une entité.
   *
   * Remplace le \a index-ème noeud de l'entité de la famille \a family()
   * de numéro local \a item_lid par le noeud de numéro local \a new_node_lid.
   */
  virtual void replaceNode(ItemLocalId item_lid, Integer index,
                           ItemLocalId new_node_lid) = 0;

  /*!
   * \brief Remplace une arête d'une entité.
   *
   * Remplace la \a index-ème arête de l'entité de la famille \a family()
   * de numéro local \a item_lid par l'arête de numéro local \a new_edge_lid.
   */
  virtual void replaceEdge(ItemLocalId item_lid, Integer index,
                           ItemLocalId new_edge_lid) = 0;

  /*!
   * \brief Remplace une face d'une entité.
   *
   * Remplace la \a index-ème face de l'entité de la famille \a family()
   * de numéro local \a item_lid par la face de numéro local \a new_face_lid.
   */
  virtual void replaceFace(ItemLocalId item_lid, Integer index,
                           ItemLocalId new_face_lid) = 0;

  /*!
   * \brief Remplace une maille d'une entité.
   *
   * Remplace la \a index-ème maille de l'entité de la famille \a family()
   * de numéro local \a item_lid par la face de numéro local \a new_cell_lid.
   */
  virtual void replaceCell(ItemLocalId item_lid, Integer index,
                           ItemLocalId new_cell_lid) = 0;

  /*!
   * \brief Remplace une entité parente d'une entité.
   *
   * Remplace la \a index-ème entité parent de l'entité de la famille \a family()
   * de numéro local \a item_lid par l'entité parent de numéro local \a new_hparent_lid.
   */
  virtual void replaceHParent(ItemLocalId item_lid, Integer index,
                              ItemLocalId new_hparent_lid) = 0;

  /*!
   * \brief Remplace une entité enfant d'une entité.
   *
   * Remplace la \a index-ème entité enfant de l'entité de la famille \a family()
   * de numéro local \a item_lid par l'entité enfant de numéro local \a new_hchild_lid.
   */
  virtual void replaceHChild(ItemLocalId item_lid, Integer index,
                             ItemLocalId new_hchild_lid) = 0;

  /*!
   * \brief Remplace un noeud d'une entité.
   *
   * Remplace le noeud de numéro local \a old_node_lid de l'entité de la famille \a family()
   * de numéro local \a item_lid par le noeud de numéro local \a new_node_lid.
   *
   * Lance une exception si le noeud \a old_node_id n'est pas trouvé.
   */
  virtual void findAndReplaceNode(ItemLocalId item_lid, ItemLocalId old_node_lid,
                                  ItemLocalId new_node_lid) = 0;

  /*!
   * \brief Remplace une arête d'une entité.
   *
   * Remplace l'arête de numéro local \a old_edge_lid de l'entité de la famille \a family()
   * de numéro local \a item_lid par l'arête de numéro local \a new_edge_lid.
   *
   * Lance une exception si l'arête \a old_edge_lid n'est pas trouvée.
   */
  virtual void findAndReplaceEdge(ItemLocalId item_lid, ItemLocalId old_edge_lid,
                                  ItemLocalId new_edge_lid) = 0;

  /*!
   * \brief Remplace une face d'une entité.
   *
   * Remplace la face de numéro local \a old_face_lid  de l'entité de la famille \a family()
   * de numéro local \a item_lid par la face de numéro local \a new_face_lid.
   *
   * Lance une exception si la face \a old_face_lid n'est pas trouvée.
   */
  virtual void findAndReplaceFace(ItemLocalId item_lid, ItemLocalId old_face_lid,
                                  ItemLocalId new_face_lid) = 0;

  /*!
   * \brief Remplace une maille d'une entité.
   *
   * Remplace la maille de numéro local \a old_cell_lid de l'entité de la famille \a family()
   * de numéro local \a item_lid par la face de numéro local \a new_cell_lid.
   *
   * Lance une exception si la maille \a old_cell_lid n'est pas trouvée.
   */
  virtual void findAndReplaceCell(ItemLocalId item_lid, ItemLocalId old_cell_lid,
                                  ItemLocalId new_cell_lid) = 0;

  /*!
   * \brief Positionne la maille devant et derrière une face.
   *
   * Cette méthode n'est implémentée que pour les familles de face. Pour les autres
   * famille, lève une exception de type NotSupportedException.
   *
   * * \param face_lid numéro local de la face
   * \param back_cell_lid numéro local de la maille derrière (ou NULL_ITEM_LOCAL_ID)
   * \param front_cell_lid numéro local de la maille devant (ou NULL_ITEM_LOCAL_ID)
   */
  virtual void setBackAndFrontCells(FaceLocalId face_lid, CellLocalId back_cell_lid,
                                    CellLocalId front_cell_lid);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
