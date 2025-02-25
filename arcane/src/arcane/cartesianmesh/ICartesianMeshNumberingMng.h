// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshNumberingMng.h                                (C) 2000-2024 */
/*                                                                           */
/* Interface de gestionnaire de numérotation pour maillage cartesian.        */
/* Dans ces gestionnaires, on considère que l'on a un intervalle des         */
/* uniqueids attribué à chaque niveau du maillage.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_ICARTESIANMESHNUMBERINGMNG_H
#define ARCANE_CARTESIANMESH_ICARTESIANMESHNUMBERINGMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshNumberingMng
{
 public:

  virtual ~ICartesianMeshNumberingMng() = default;

 public:

  /*!
   * \brief Méthode permettant de préparer un nouveau niveau.
   *
   * Avant de raffiner ou de déraffiner des mailles, il est
   * nécessaire d'appeler cette méthode pour préparer l'objet
   * à fournir les informations concernant le nouveau niveau.
   *
   * Il faut aussi noter que ce nouveau niveau doit être le niveau
   * directement supérieur au plus haut niveau déjà existant ou
   * directement inférieur au plus bas niveau déjà existant.
   *
   * \param level Le nouveau niveau à préparer.
   */
  virtual void prepareLevel(Int32 level) =0;

  /*!
   * \brief Méthode permettant de mettre à jour le premier niveau.
   *
   * En effet, lors du déraffinement, le nouveau niveau est le
   * niveau -1. Arcane n'appréciant pas les niveaux négatifs,
   * on doit mettre à jour les informations pour ne plus en avoir.
   */
  virtual void updateFirstLevel() = 0;

  /*!
   * \brief Méthode permettant de récupérer le premier unique id utilisé par les mailles d'un niveau.
   * L'appel de cette méthode avec level et level+1 permet de récupérer l'intervalle des uniqueids
   * d'un niveau.
   *
   * \param level Le niveau.
   * \return Le premier uid des mailles du niveau.
   */
  virtual Int64 firstCellUniqueId(Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer le premier unique id utilisé par les noeuds d'un niveau.
   * L'appel de cette méthode avec level et level+1 permet de récupérer l'intervalle des uniqueids
   * d'un niveau.
   *
   * \param level Le niveau.
   * \return Le premier uid des noeuds du niveau.
   */
  virtual Int64 firstNodeUniqueId(Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer le premier unique id utilisé par les faces d'un niveau.
   * L'appel de cette méthode avec level et level+1 permet de récupérer l'intervalle des uniqueids
   * d'un niveau.
   *
   * \param level Le niveau.
   * \return Le premier uid des faces du niveau.
   */
  virtual Int64 firstFaceUniqueId(Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de mailles global en X d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de mailles en X.
   */
  virtual Int64 globalNbCellsX(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de mailles global en Y d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de mailles en Y.
   */
  virtual Int64 globalNbCellsY(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de mailles global en Z d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de mailles en Z.
   */
  virtual Int64 globalNbCellsZ(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de noeuds global en X d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de noeuds en X.
   */
  virtual Int64 globalNbNodesX(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de noeuds global en Y d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de noeuds en Y.
   */
  virtual Int64 globalNbNodesY(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de noeuds global en Z d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de noeuds en Z.
   */
  virtual Int64 globalNbNodesZ(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de faces global en X d'un niveau.
   *
   * Admettons que l'on ai les faces suivantes :
   *  ┌─0──┬──2─┐
   * 4│   6│   8│
   *  ├─5──┼─7──┤
   * 9│  11│  13│
   *  └─10─┴─12─┘
   *
   * Donc, on a 2x2 mailles.
   * En X, on a 3 faces.
   *
   * Pour le nombre de faces en vue cartésienne, voir \a globalNbFacesXCartesianView.
   *
   * \param level Le niveau.
   * \return Le nombre de faces en X.
   */
  virtual Int64 globalNbFacesX(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de faces global en Y d'un niveau.
   *
   * Admettons que l'on ai les faces suivantes :
   *  ┌─0──┬──2─┐
   * 4│   6│   8│
   *  ├─5──┼─7──┤
   * 9│  11│  13│
   *  └─10─┴─12─┘
   *
   * Donc, on a 2x2 mailles.
   * En Y, on a 3 faces.
   *
   * Pour le nombre de faces en vue cartésienne, voir \a globalNbFacesYCartesianView.
   *
   * \param level Le niveau.
   * \return Le nombre de faces en Y.
   */
  virtual Int64 globalNbFacesY(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de faces global en Z d'un niveau.
   *
   * Admettons que l'on ai les faces suivantes :
   *  ┌─0──┬──2─┐
   * 4│   6│   8│
   *  ├─5──┼─7──┤
   * 9│  11│  13│
   *  └─10─┴─12─┘
   *
   * Si on a 2x2x2 mailles, on aura, en Z, 3 faces.
   *
   * Pour le nombre de faces en vue cartésienne, voir \a globalNbFacesZCartesianView.
   *
   * \param level Le niveau.
   * \return Le nombre de faces en Z.
   */
  virtual Int64 globalNbFacesZ(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer la taille de la vue "grille cartésienne"
   *        contenant les noeuds.
   *
   * En 2D, on peut avoir cette vue :
   *      x =  0  1  2  3  4
   *         ┌──┬──┬──┬──┬──┐
   *  y = 0  │  │ 1│  │ 3│  │
   *         ├──┼──┼──┼──┼──┤
   *  y = 1  │ 5│  │ 7│  │ 9│
   *         ├──┼──┼──┼──┼──┤
   *  y = 2  │  │ 6│  │ 8│  │
   *         ├──┼──┼──┼──┼──┤
   *  y = 3  │10│  │12│  │14│
   *         ├──┼──┼──┼──┼──┤
   *  y = 4  │  │11│  │13│  │
   *         └──┴──┴──┴──┴──┘
   *
   * Et en 3D :
   *         z = 0            │ z = 1            │ z = 2            │ z = 3            │ z = 4
   *      x =  0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4
   *         ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐
   *  y = 0  │  │  │  │  │  │ │ │  │24│  │25│  │ │ │  │  │  │  │  │ │ │  │30│  │31│  │ │ │  │  │  │  │  │
   *         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
   *  y = 1  │  │ 0│  │ 1│  │ │ │12│  │13│  │14│ │ │  │ 4│  │ 5│  │ │ │18│  │19│  │20│ │ │  │ 8│  │ 9│  │
   *         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
   *  y = 2  │  │  │  │  │  │ │ │  │26│  │27│  │ │ │  │  │  │  │  │ │ │  │32│  │33│  │ │ │  │  │  │  │  │
   *         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
   *  y = 3  │  │ 2│  │ 3│  │ │ │15│  │16│  │17│ │ │  │ 6│  │ 7│  │ │ │21│  │22│  │23│ │ │  │10│  │11│  │
   *         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
   *  y = 4  │  │  │  │  │  │ │ │  │28│  │29│  │ │ │  │  │  │  │  │ │ │  │34│  │35│  │ │ │  │  │  │  │  │
   *         └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘
   *                          │                  │                  │                  │
   *
   * \param level Le niveau.
   * \return La taille de la grille en X.
   */
  virtual Int64 globalNbFacesXCartesianView(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer la taille de la vue "grille cartésienne"
   *        contenant les noeuds.
   *
   * Un exemple de cette vue est disponible dans la documentation de \a globalNbFacesXCartesianView.
   *
   * \param level Le niveau.
   * \return La taille de la grille en Y.
   */
  virtual Int64 globalNbFacesYCartesianView(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer la taille de la vue "grille cartésienne"
   *        contenant les noeuds.
   *
   * Un exemple de cette vue est disponible dans la documentation de \a globalNbFacesXCartesianView.
   *
   * \param level Le niveau.
   * \return La taille de la grille en Z.
   */
  virtual Int64 globalNbFacesZCartesianView(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de mailles total dans un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de mailles dans le niveau.
   */
  virtual Int64 nbCellInLevel(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de noeuds total dans un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de noeuds dans le niveau.
   */
  virtual Int64 nbNodeInLevel(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de faces total dans un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de faces dans le niveau.
   */
  virtual Int64 nbFaceInLevel(Integer level) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le pattern de raffinement utilisé dans chaque maille.
   * Par exemple, si le pattern vaut 2, chaque maille parente aura 2*2 mailles filles (2*2*2 en 3D).
   *
   * \return Le pattern de raffinement.
   */
  virtual Integer pattern() const = 0;

  /*!
   * \brief Méthode permettant de récupérer le niveau d'une maille avec son uid.
   *
   * \param uid L'uniqueId de la maille.
   * \return Le niveau de la maille.
   */
  virtual Int32 cellLevel(Int64 uid) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le niveau d'un noeud avec son uid.
   *
   * \param uid L'uniqueId du noeud.
   * \return Le niveau du noeud.
   */
  virtual Int32 nodeLevel(Int64 uid) const = 0;

  /*!
   * \brief Méthode permettant de récupérer le niveau d'une face avec son uid.
   *
   * \param uid L'uniqueId de la face.
   * \return Le niveau de la face.
   */
  virtual Int32 faceLevel(Int64 uid) const = 0;

  /*!
   * \brief Méthode permettant d'obtenir la position du premier noeud/maille fille à partir de la position
   * du noeud/maille parent.
   *
   * Exemple : si l'on a un maillage 2D de 2*2 mailles et un pattern de raffinement de 2,
   * on sait que la grille de niveau 1 (pour les patchs de niveau 1) sera de 4*4 mailles.
   * Le premier noeud/maille fille du noeud/maille parent (Xp=1,Yp=0) aura la position Xf=Xp*Pattern=2 (idem pour Y).
   *
   * \param coord La position X ou Y ou Z du noeud/maille parent.
   * \param level_from Le niveau parent.
   * \param level_to Le niveau enfant.
   * \return La position de la première fille du noeud/maille parent.
   */
  virtual Int64 offsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const = 0;

  /*!
   * \brief Méthode permettant d'obtenir la position de la première face enfant à partir de la position
   * de la face parente.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param coord La position X ou Y ou Z de la face parente.
   * \param level_from Le niveau parent.
   * \param level_to Le niveau enfant.
   * \return La position du premier enfant de la face parente.
   */
  virtual Int64 faceOffsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'une maille grâce à son uniqueId.
   *
   * \param uid L'uniqueId de la maille.
   * \param level Le niveau de la maille.
   * \return La position en X de la maille.
   */
  virtual Int64 cellUniqueIdToCoordX(Int64 uid, Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'une maille.
   *
   * \param cell La maille.
   * \return La position en X de la maille.
   */
  virtual Int64 cellUniqueIdToCoordX(Cell cell) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'une maille grâce à son uniqueId.
   *
   * \param uid L'uniqueId de la maille.
   * \param level Le niveau de la maille.
   * \return La position en Y de la maille.
   */
  virtual Int64 cellUniqueIdToCoordY(Int64 uid, Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'une maille.
   *
   * \param cell La maille.
   * \return La position en Y de la maille.
   */
  virtual Int64 cellUniqueIdToCoordY(Cell cell) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'une maille grâce à son uniqueId.
   *
   * \param uid L'uniqueId de la maille.
   * \param level Le niveau de la maille.
   * \return La position en Z de la maille.
   */
  virtual Int64 cellUniqueIdToCoordZ(Int64 uid, Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'une maille.
   *
   * \param cell La maille.
   * \return La position en Z de la maille.
   */
  virtual Int64 cellUniqueIdToCoordZ(Cell cell) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'un noeud grâce à son uniqueId.
   *
   * \param uid L'uniqueId du noeud.
   * \param level Le niveau du noeud.
   * \return La position en X du noeud.
   */
  virtual Int64 nodeUniqueIdToCoordX(Int64 uid, Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'un noeud.
   *
   * \param node Le noeud.
   * \return La position en X du noeud.
   */
  virtual Int64 nodeUniqueIdToCoordX(Node node) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'un noeud grâce à son uniqueId.
   *
   * \param uid L'uniqueId du noeud.
   * \param level Le niveau du noeud.
   * \return La position en Y du noeud.
   */
  virtual Int64 nodeUniqueIdToCoordY(Int64 uid, Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'un noeud.
   *
   * \param node Le noeud.
   * \return La position en Y du noeud.
   */
  virtual Int64 nodeUniqueIdToCoordY(Node node) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'un noeud grâce à son uniqueId.
   *
   * \param uid L'uniqueId du noeud.
   * \param level Le niveau du noeud.
   * \return La position en Z du noeud.
   */
  virtual Int64 nodeUniqueIdToCoordZ(Int64 uid, Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'un noeud.
   *
   * \param node Le noeud.
   * \return La position en Z du noeud.
   */
  virtual Int64 nodeUniqueIdToCoordZ(Node node) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'une face grâce à son uniqueId.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param uid L'uniqueId de la face.
   * \param level Le niveau de la face.
   * \return La position en X de la face.
   */
  virtual Int64 faceUniqueIdToCoordX(Int64 uid, Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'une face.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param face La face.
   * \return La position en X de la face.
   */
  virtual Int64 faceUniqueIdToCoordX(Face face) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'une face grâce à son uniqueId.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param uid L'uniqueId de la face.
   * \param level Le niveau de la face.
   * \return La position en Y de la face.
   */
  virtual Int64 faceUniqueIdToCoordY(Int64 uid, Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'une face.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param face La face.
   * \return La position en Y de la face.
   */
  virtual Int64 faceUniqueIdToCoordY(Face face) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'une face grâce à son uniqueId.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param uid L'uniqueId de la face.
   * \param level Le niveau de la face.
   * \return La position en Z de la face.
   */
  virtual Int64 faceUniqueIdToCoordZ(Int64 uid, Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'une face.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param face La face.
   * \return La position en Z de la face.
   */
  virtual Int64 faceUniqueIdToCoordZ(Face face) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une maille à partir de sa position et de son niveau.
   *
   * \param level Le niveau de la maille.
   * \param cell_coord La position de la maille.
   * \return L'uniqueId de la maille.
   */
  virtual Int64 cellUniqueId(Integer level, Int64x3 cell_coord) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une maille à partir de sa position et de son niveau.
   *
   * \param level Le niveau de la maille.
   * \param cell_coord La position de la maille.
   * \return L'uniqueId de la maille.
   */
  virtual Int64 cellUniqueId(Integer level, Int64x2 cell_coord) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'un noeud à partir de sa position et de son niveau.
   *
   * \param level Le niveau du noeud.
   * \param cell_coord La position du noeud.
   * \return L'uniqueId du noeud.
   */
  virtual Int64 nodeUniqueId(Integer level, Int64x3 node_coord) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'un noeud à partir de sa position et de son niveau.
   *
   * \param level Le niveau du noeud.
   * \param cell_coord La position du noeud.
   * \return L'uniqueId du noeud.
   */
  virtual Int64 nodeUniqueId(Integer level, Int64x2 node_coord) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une face à partir de sa position et de son niveau.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param level Le niveau de la face.
   * \param cell_coord La position de la face.
   * \return L'uniqueId de la face.
   */
  virtual Int64 faceUniqueId(Integer level, Int64x3 face_coord) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une face à partir de sa position et de son niveau.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param level Le niveau de la face.
   * \param cell_coord La position de la face.
   * \return L'uniqueId de la face.
   */
  virtual Int64 faceUniqueId(Integer level, Int64x2 face_coord) = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de noeuds dans une maille.
   *
   * \return Le nombre de noeuds d'une maille.
   */
  virtual Integer nbNodeByCell() = 0;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des noeuds d'une maille à partir de
   * ses coordonnées.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des noeuds
   * d'une maille d'Arcane.
   *      3--2
   * ^y   |  |
   * |    0--1
   * ->x
   *
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbNodeByCell().
   * \param level Le niveau de la maille (et donc des noeuds).
   * \param cell_coord La position de la maille.
   */
  virtual void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord) = 0;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des noeuds d'une maille à partir de
   * ses coordonnées.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des noeuds
   * d'une maille d'Arcane.
   *      3--2
   * ^y   |  |
   * |    0--1
   * ->x
   *
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbNodeByCell().
   * \param level Le niveau de la maille (et donc des noeuds).
   * \param cell_coord La position de la maille.
   */
  virtual void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord) = 0;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des noeuds d'une maille à partir de
   * son uniqueId.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des noeuds
   * d'une maille d'Arcane.
   *      3--2
   * ^y   |  |
   * |    0--1
   * ->x
   *
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbNodeByCell().
   * \param level Le niveau de la maille (et donc des noeuds).
   * \param cell_uid L'uniqueId de la maille.
   */
  virtual void cellNodeUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid) = 0;

  /*!
   * \brief Méthode permettant de récupérer le nombre de faces dans une maille.
   *
   * \return Le nombre de faces d'une maille.
   */
  virtual Integer nbFaceByCell() = 0;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des faces d'une maille à partir de
   * ses coordonnées.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des faces
   * d'une maille d'Arcane.
   *      -2-
   * ^y   3 1
   * |    -0-
   * ->x
   *
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbFaceByCell().
   * \param level Le niveau de la maille (et donc des faces).
   * \param cell_coord La position de la maille.
   */
  virtual void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x3 cell_coord) = 0;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des faces d'une maille à partir de
   * ses coordonnées.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des faces
   * d'une maille d'Arcane.
   *      -2-
   * ^y   3 1
   * |    -0-
   * ->x
   *
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbFaceByCell().
   * \param level Le niveau de la maille (et donc des faces).
   * \param cell_coord La position de la maille.
   */
  virtual void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64x2 cell_coord) = 0;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des faces d'une maille à partir de
   * son uniqueId.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des faces
   * d'une maille d'Arcane.
   *      -2-
   * ^y   3 1
   * |    -0-
   * ->x
   *
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbFaceByCell().
   * \param level Le niveau de la maille (et donc des faces).
   * \param cell_uid L'uniqueId de la maille.
   */
  virtual void cellFaceUniqueIds(ArrayView<Int64> uid, Integer level, Int64 cell_uid) = 0;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des mailles autour de la maille passée
   * en paramètre.
   *
   * La vue passée en paramètre doit faire une taille de 9 en 2D et de 27 en 3D.
   *
   * \param uid [OUT] Les uniqueIds des mailles autour.
   * \param cell_uid L'uniqueId de la maille au centre.
   * \param level Le niveau de la maille au centre.
   */
  virtual void cellUniqueIdsAroundCell(ArrayView<Int64> uid, Int64 cell_uid, Int32 level) = 0;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des mailles autour de la maille passée
   * en paramètre.
   *
   * La vue passée en paramètre doit faire une taille de 9 en 2D et de 27 en 3D.
   *
   * \param uid [OUT] Les uniqueIds des mailles autour.
   * \param cell La maille au centre.
   */
  virtual void cellUniqueIdsAroundCell(ArrayView<Int64> uid, Cell cell) = 0;

  /*!
   * \brief Méthode permettant de définir les coordonnées spatiales des noeuds des mailles enfants
   * d'une maille parent.
   * Cette méthode doit être appelée après l'appel à endUpdate().
   *
   * \param parent_cell La maille parent.
   */
  virtual void setChildNodeCoordinates(Cell parent_cell) = 0;

  /*!
   * \brief Méthode permettant de définir les coordonnées spatiales des noeuds d'une maille parent.
   * Cette méthode doit être appelée après l'appel à endUpdate().
   *
   * \param parent_cell La maille parent.
   */
  virtual void setParentNodeCoordinates(Cell parent_cell) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId du parent d'une maille.
   *
   * Si \a do_fatal est vrai, une erreur fatale est générée si le parent n'existe
   * pas, sinon l'uniqueId retourné a pour valeur NULL_ITEM_UNIQUE_ID.
   *
   * \param uid L'uniqueId de la maille enfant.
   * \param level Le niveau de la maille enfant.
   * \return L'uniqueId de la maille parent de la maille passé en paramètre.
   */
  virtual Int64 parentCellUniqueIdOfCell(Int64 uid, Integer level, bool do_fatal = true) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId du parent d'une maille.
   *
   * Si \a do_fatal est vrai, une erreur fatale est générée si le parent n'existe
   * pas, sinon l'uniqueId retourné a pour valeur NULL_ITEM_UNIQUE_ID.
   *
   * \param cell La maille enfant.
   * \return L'uniqueId de la maille parent de la maille passé en paramètre.
   */
  virtual Int64 parentCellUniqueIdOfCell(Cell cell, bool do_fatal = true) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une maille enfant d'une maille parent
   * à partir de la position de la maille enfant dans la maille parent.
   *
   * \param cell La maille parent.
   * \param child_coord_in_parent La position de l'enfant dans la maille parent.
   * \return L'uniqueId de la maille enfant demandée.
   */
  virtual Int64 childCellUniqueIdOfCell(Cell cell, Int64x3 child_coord_in_parent) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une maille enfant d'une maille parent
   * à partir de la position de la maille enfant dans la maille parent.
   *
   * \param cell La maille parent.
   * \param child_coord_in_parent La position de l'enfant dans la maille parent.
   * \return L'uniqueId de la maille enfant demandée.
   */
  virtual Int64 childCellUniqueIdOfCell(Cell cell, Int64x2 child_coord_in_parent) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une maille enfant d'une maille parent
   * à partir de l'index de la maille enfant dans la maille parent.
   *
   * \param cell La maille parent.
   * \param child_index_in_parent L'index de l'enfant dans la maille parent.
   * \return L'uniqueId de la maille enfant demandée.
   */
  virtual Int64 childCellUniqueIdOfCell(Cell cell, Int64 child_index_in_parent) = 0;

  /*!
   * \brief Méthode permettant de récupérer une maille enfant d'une maille parent
   * à partir de la position de la maille enfant dans la maille parent.
   *
   * \param cell La maille parent.
   * \param child_coord_in_parent La position de l'enfant dans la maille parent.
   * \return La maille enfant demandée.
   */
  virtual Cell childCellOfCell(Cell cell, Int64x3 child_coord_in_parent) = 0;

  /*!
   * \brief Méthode permettant de récupérer une maille enfant d'une maille parent
   * à partir de la position de la maille enfant dans la maille parent.
   *
   * \param cell La maille parent.
   * \param child_coord_in_parent La position de l'enfant dans la maille parent.
   * \return La maille enfant demandée.
   */
  virtual Cell childCellOfCell(Cell cell, Int64x2 child_coord_in_parent) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId du parent d'un noeud.
   *
   * Si \a do_fatal est vrai, une erreur fatale est générée si le parent n'existe
   * pas, sinon l'uniqueId retourné a pour valeur NULL_ITEM_UNIQUE_ID.
   *
   * \param uid L'uniqueId du noeud enfant.
   * \param level Le niveau du noeud enfant.
   * \return L'uniqueId du noeud parent du noeud enfant.
   */
  virtual Int64 parentNodeUniqueIdOfNode(Int64 uid, Integer level, bool do_fatal = true) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId du parent d'un noeud.
   *
   * Si \a do_fatal est vrai, une erreur fatale est générée si le parent n'existe
   * pas, sinon l'uniqueId retourné a pour valeur NULL_ITEM_UNIQUE_ID.
   *
   * \param node Le noeud enfant.
   * \return L'uniqueId du noeud parent du noeud passé en paramètre.
   */
  virtual Int64 parentNodeUniqueIdOfNode(Node node, bool do_fatal = true) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'un noeud enfant d'un noeud parent.
   *
   * \param uid L'uniqueId du noeud enfant.
   * \param level Le niveau du noeud enfant.
   * \return L'uniqueId du noeud enfant demandée.
   */
  virtual Int64 childNodeUniqueIdOfNode(Int64 uid, Integer level) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'un noeud enfant d'un noeud parent.
   *
   * \param node Le noeud parent.
   * \return L'uniqueId du noeud enfant demandée.
   */
  virtual Int64 childNodeUniqueIdOfNode(Node node) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId du parent d'une face.
   *
   * Si \a do_fatal est vrai, une erreur fatale est générée si le parent n'existe
   * pas, sinon l'uniqueId retourné a pour valeur NULL_ITEM_UNIQUE_ID.
   *
   * \param uid L'uniqueId de la face enfant.
   * \param level Le niveau de la face enfant.
   * \return L'uniqueId de la face parent de la face enfant.
   */
  virtual Int64 parentFaceUniqueIdOfFace(Int64 uid, Integer level, bool do_fatal = true) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId du parent d'une face.
   *
   * Si \a do_fatal est vrai, une erreur fatale est générée si le parent n'existe
   * pas, sinon l'uniqueId retourné a pour valeur NULL_ITEM_UNIQUE_ID.
   *
   * \param face La face enfant.
   * \return L'uniqueId de la face parent de la face passé en paramètre.
   */
  virtual Int64 parentFaceUniqueIdOfFace(Face face, bool do_fatal = true) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une face enfant d'une face parent
   * à partir de l'index de la face enfant dans la face parent.
   *
   * \param uid L'uniqueId de la face parent.
   * \param level Le niveau de la face parent.
   * \param child_index_in_parent L'index de l'enfant dans la face parent.
   * \return L'uniqueId de la face enfant demandée.
   */
  virtual Int64 childFaceUniqueIdOfFace(Int64 uid, Integer level, Int64 child_index_in_parent) = 0;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une face enfant d'une face parent
   * à partir de l'index de la face enfant dans la face parent.
   *
   * \param face La face parent.
   * \param child_index_in_parent L'index de l'enfant dans la face parent.
   * \return L'uniqueId de la face enfant demandée.
   */
  virtual Int64 childFaceUniqueIdOfFace(Face face, Int64 child_index_in_parent) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_ICARTESIANMESHNUMBERINGMNG_H
