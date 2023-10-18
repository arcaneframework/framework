// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshModifier.h                                             (C) 2000-2022 */
/*                                                                           */
/* Interface de modification du maillage.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHMODIFIER_H
#define ARCANE_IMESHMODIFIER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class IExtraGhostCellsBuilder;
class IExtraGhostParticlesBuilder;
class IAMRTransportFunctor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface de modification du maillage.
 *
 * Cette interface fournit les services permettant de modifier un
 * maillage.
 */
class IMeshModifier
{
 public:

  virtual ~IMeshModifier() {} //<! Libère les ressources

 public:

  virtual void build() =0;

 public:

  //! Maillage associé
  virtual IMesh* mesh() =0;

 public:

  /*!
   * \brief Positionne la propriété indiquant si le maillage peut évoluer.
   *
   * Cette propriété doit être positionnée à vrai si l'on souhaite modifier
   * le maillage, par exemple en échangeant des entités par appel à
   * exchangeItems(). Cela ne concerne que les noeuds, arêtes, faces et
   * mailles mais pas les particules qui peuvent toujours être créées et détruites.
   *
   * Par défaut, isDynamic() est faux.
   *
   * Le positionnement de la propriété ne peut se faire qu'à l'initialisation.
   */
  virtual void setDynamic(bool v) =0;

  /*!
   * \brief Ajoute des mailles.
   *
   * Ajoute des mailles. Le format de \a cells_infos est identiques à celui
   * de la méthode IMesh::allocateCells(). Si \a cells_lid n'est pas vide, il contiendra
   * en retour les numéros locaux des mailles créées. Il est possible de faire plusieurs ajouts
   * successifs. Une fois les ajouts terminés, il faut appeler la méthode
   * endUpdate(). Si une maille ajoutée possède le même uniqueId() qu'une
   * des mailles existantes, la maille existante est conservée telle qu'elle et
   * rien ne se passe.
   *
   * Les mailles créées sont considérées comme appartenant à ce sous-domaine
   * Si ce n'est pas le cas, il faut ensuite modifier leur appartenance.
   *
   * Cette méthode est collective. Si un sous-domaine ne souhaite pas ajouter
   * de mailles, il est possible de passer un tableau vide.
   */
  virtual void addCells(Integer nb_cell,Int64ConstArrayView cell_infos,
                        Int32ArrayView cells_lid = Int32ArrayView()) =0;


  /*!
   * \brief Ajoute des faces.
   *
   * Ajoute des faces. Le format de \a face_infos est identiques à celui
   * de la méthode IMesh::allocateCells(). Si \a face_lids n'est pas vide, il contiendra
   * en retour les numéros locaux des faces créées. Il est possible de faire plusieurs ajouts
   * successifs. Une fois les ajouts terminés, il faut appeler la méthode
   * endUpdate(). Si une face ajoutée possède le même uniqueId() qu'une
   * des faces existantes, la face existante est conservée telle qu'elle et
   * rien ne se passe.
   *
   * Les faces créées sont considérées comme appartenant à ce sous-domaine
   * Si ce n'est pas le cas, il faut ensuite modifier leur appartenance.
   *
   * Cette méthode est collective. Si un sous-domaine ne souhaite pas ajouter
   * de faces, il est possible de passer un tableau vide.
   */
  virtual void addFaces(Integer nb_face,Int64ConstArrayView face_infos,
                        Int32ArrayView face_lids = Int32ArrayView()) =0;

  /*!
   * \brief Ajoute des arêtes.
   *
   * Ajoute des arêtes. Le format de \a edge_infos est identiques à celui
   * de la méthode IMesh::allocateCells(). Si \a edge_lids n'est pas vide, il contiendra
   * en retour les numéros locaux des arêtes créées. Il est possible de faire plusieurs ajouts
   * successifs. Une fois les ajouts terminés, il faut appeler la méthode
   * endUpdate(). Si une face ajoutée possède le même uniqueId() qu'une
   * des arêtes existantes, la arête existante est conservée telle qu'elle et
   * rien ne se passe.
   *
   * Les arêtes créées sont considérées comme appartenant à ce sous-domaine
   * Si ce n'est pas le cas, il faut ensuite modifier leur appartenance.
   *
   * Cette méthode est collective. Si un sous-domaine ne souhaite pas ajouter
   * de arêtes, il est possible de passer un tableau vide.
   */
  virtual void addEdges(Integer nb_edge,Int64ConstArrayView edge_infos,
                        Int32ArrayView edge_lids = Int32ArrayView()) =0;

  /*!
   * \brief Ajoute des noeuds.
   *
   * Ajoute des noeuds avec comme identifiant unique les valeurs
   * du tableau \a nodes_uid. Si \a nodes_lid n'est pas vide, il contiendra
   * en retour les numéros locaux des noeuds créés. Il est possible de faire plusieurs ajouts
   * successifs. Une fois les ajouts terminés, il faut appeler la méthode
   * endUpdate(). Il est possible de spécifier un uniqueId() déjà
   * existant. Dans ce cas le noeud est simplement ignoré.
   *
   * Les noeuds créés sont considérés comme appartenant à ce sous-domaine
   * Si ce n'est pas le cas, il faut ensuite modifier leur appartenance.
   *
   * Cette méthode est collective. Si un sous-domaine ne souhaite pas ajouter
   * de noeuds, il est possible de passer un tableau vide.
   */
  virtual void addNodes(Int64ConstArrayView nodes_uid,
                        Int32ArrayView nodes_lid = Int32ArrayView()) =0;

  /*!
   * \brief Supprime des mailles.
   *
   * Supprime les mailles dont les numéros locaux sont données
   * dans \a cells_local_id. Il est possible de faire plusieurs suppressions
   * successives. Une fois les suppressions terminées, il faut appeler la méthode
   * endUpdate().
   */
  virtual void removeCells(Int32ConstArrayView cells_local_id) =0;

  virtual void removeCells(Int32ConstArrayView cells_local_id,bool update_ghost) = 0 ;

  /*!
   * \brief Détache des mailles du maillage.
   *
   * Les mailles détachées sont déconnectées du maillage. Les noeuds, arêtes et faces
   * de ces mailles ne leur font plus référence et le uniqueId() de ces mailles peuvent
   * être réutilisés. Pour détruire définitivement ces mailles, il faut appeler
   * la méthode removeDetachedCells().
   */
  virtual void detachCells(Int32ConstArrayView cells_local_id) =0;

  /*!
   * \brief Supprime les mailles détachées
   *
   * Supprime les mailles détachées via detachCells().
   * Il est possible de faire plusieurs suppressions
   * successives. Une fois les suppressions terminées, il faut appeler la méthode
   * endUpdate().
   */
  virtual void removeDetachedCells(Int32ConstArrayView cells_local_id) =0;

  //! AMR
  virtual void flagCellToRefine(Int32ConstArrayView cells_lids) =0;
  virtual void flagCellToCoarsen(Int32ConstArrayView cells_lids) =0;
  virtual void refineItems() =0;
  virtual void coarsenItems() =0;
  virtual bool adapt() =0;
  virtual void registerCallBack(IAMRTransportFunctor* f) =0;
  virtual void unRegisterCallBack(IAMRTransportFunctor* f) =0;
  virtual void addHChildrenCells(Cell parent_cell,Integer nb_cell,
                                 Int64ConstArrayView cells_infos,Int32ArrayView cells_lid = Int32ArrayView()) =0;
  virtual void addParentCellToCell(Cell child, Cell parent) = 0;
  virtual void addChildCellToCell(Cell parent, Cell child) = 0;

  //! Supprime toutes les entitées de toutes les familles de ce maillage.
  virtual void clearItems() =0;

  /*!
   * \brief Ajoute les mailles à partir des données contenues dans \a buffer.
   *
   * \a buffer doit contenir des mailles sérialisées, par exemple par
   * l'appel à IMesh::serializeCells().
   *
   * \deprecated Utiliser IMesh::cellFamily()->policyMng()->createSerializer() à la place.
   */
  ARCANE_DEPRECATED_240 virtual void addCells(ISerializer* buffer) =0;

  /*!
   * \brief Ajoute les mailles à partir des données contenues dans \a buffer.
   *
   * \a buffer doit contenir des mailles sérialisées, par exemple par
   * l'appel à IMesh::serializeCells().
   * En retour \a cells_local_id contient la liste des localId() des
   * mailles désérialisées. Une maille peut être présente plusieurs
   * fois dans cette liste si elle est présente plusieurs fois dans \a buffer.
   *
   * \deprecated Utiliser IMesh::cellFamily()->policyMng()->createSerializer() à la place.
   */
  ARCANE_DEPRECATED_240 virtual void addCells(ISerializer* buffer,Int32Array& cells_local_id) =0;

  /*!
   * \brief Notifie l'instance de la fin de la modification du maillage.
   *
   * Cette méthode est collective.
   */
  virtual void endUpdate() =0;

  virtual void endUpdate(bool update_ghost_layer,bool remove_old_ghost) = 0; // SDC: this signature is needed @IFPEN.

 public:

  /*!
   * \brief Mise à jour de la couche fantôme.
   *
   * Cette opération est collective.
   */
  virtual void updateGhostLayers() =0;

  //! AMR
  virtual void updateGhostLayerFromParent(Array<Int64>& ghost_cell_to_refine,
                                          Array<Int64>& ghost_cell_to_coarsen,
                                          bool remove_old_ghost) =0;

  //! ajout du algorithme d'ajout de mailles fantômes "extraordinaires".
  virtual void addExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder) =0;

  //! Supprime l'association à l'instance \a builder.
  virtual void removeExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder) =0;

  //! Ajout du algorithme d'ajout de particules fantômes "extraordinaires"
  virtual void addExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder) =0;

  //! Supprime l'association à l'instance \a builder.
  virtual void removeExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder) =0;
  
 public:

  //! Fusionne les maillages de \a meshes avec le maillage actuel.
  virtual void mergeMeshes(ConstArrayView<IMesh*> meshes) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
