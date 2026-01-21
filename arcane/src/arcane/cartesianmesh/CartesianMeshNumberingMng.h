// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshNumberingMng.h                                 (C) 2000-2026 */
/*                                                                           */
/* Gestionnaire de numérotation de maillage cartesian. La numérotation       */
/* utilisée ici est la même que celle utilisée dans la renumérotation V2.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include "arcane/utils/Ref.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICartesianMeshNumberingMngInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface de gestionnaire de numérotation pour maillage cartesian.
 *
 * Dans ces gestionnaires, on considère que l'on a un intervalle d'uniqueIds
 * attribué à chaque niveau du maillage.
 *
 * \warning Le maillage ne doit pas être renuméroté si cette numérotation est
 * utilisée.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshNumberingMng
{
 public:

  explicit CartesianMeshNumberingMng(ICartesianMesh* mesh);

 public:

  /*!
   * \brief Méthode permettant de décrire l'état de l'objet.
   */
  void printStatus() const;

  /*!
   * \brief Méthode permettant de récupérer le premier unique id utilisé par les mailles d'un niveau.
   * L'appel de cette méthode avec level et level+1 permet de récupérer l'intervalle des uniqueids
   * d'un niveau.
   *
   * \param level Le niveau.
   * \return Le premier uid des mailles du niveau.
   */
  Int64 firstCellUniqueId(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le premier unique id utilisé par les noeuds d'un niveau.
   * L'appel de cette méthode avec level et level+1 permet de récupérer l'intervalle des uniqueids
   * d'un niveau.
   *
   * \param level Le niveau.
   * \return Le premier uid des noeuds du niveau.
   */
  Int64 firstNodeUniqueId(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le premier unique id utilisé par les faces d'un niveau.
   * L'appel de cette méthode avec level et level+1 permet de récupérer l'intervalle des uniqueids
   * d'un niveau.
   *
   * \param level Le niveau.
   * \return Le premier uid des faces du niveau.
   */
  Int64 firstFaceUniqueId(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de mailles global en X d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de mailles en X.
   */
  CartCoord globalNbCellsX(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de mailles global en Y d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de mailles en Y.
   */
  CartCoord globalNbCellsY(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de mailles global en Z d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de mailles en Z.
   */
  CartCoord globalNbCellsZ(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de noeuds global en X d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de noeuds en X.
   */
  CartCoord globalNbNodesX(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de noeuds global en Y d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de noeuds en Y.
   */
  CartCoord globalNbNodesY(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de noeuds global en Z d'un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de noeuds en Z.
   */
  CartCoord globalNbNodesZ(Int32 level) const;

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
  CartCoord globalNbFacesX(Int32 level) const;

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
  CartCoord globalNbFacesY(Int32 level) const;

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
  CartCoord globalNbFacesZ(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la taille de la vue "grille cartésienne"
   *        contenant les faces.
   *
   * En 2D, on peut avoir cette vue (pour un maillage de 2x2 mailles) :
   *     x =  0  1  2  3  4
   *        ┌──┬──┬──┬──┬──┐
   * y = -1 │ 0│  │ 2│  │ 4│
   *        ┌──┬──┬──┬──┬──┐
   * y  │  │ 1│  │ 3│  │
   *        ├──┼──┼──┼──┼──┤
   * y = 1  │ 5│  │ 7│  │ 9│
   *        ├──┼──┼──┼──┼──┤
   * y = 2  │  │ 6│  │ 8│  │
   *        ├──┼──┼──┼──┼──┤
   * y = 3  │10│  │12│  │14│
   *        ├──┼──┼──┼──┼──┤
   * y = 4  │  │11│  │13│  │
   *        └──┴──┴──┴──┴──┘
   * (dans cette vue, les mailles se situent aux X et Y impaires
   * (donc ici, [1, 1], [3, 1], [1, 3] et [3, 3])).
   *
   * \note En 2D, on considère que l'on a un niveau imaginaire y=-1.
   * \warning Afin de commencer la numérotation à 0, dans les méthodes
   * retournant un uniqueId de face 2D, on fait FaceUID-1.
   *
   * Et en 3D (pour un maillage de 2x2x2 mailles) :
   *         z            │ z = 1            │ z = 2            │ z = 3            │ z = 4
   *      x =  0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4
   *         ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐
   *  y  │  │  │  │  │  │ │ │  │24│  │25│  │ │ │  │  │  │  │  │ │ │  │30│  │31│  │ │ │  │  │  │  │  │
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
   * (dans cette vue, les mailles se situent aux X, Y et Z impaires
   * (donc ici, [1, 1, 1], [3, 1, 1], [1, 3, 1], &c)).
   *
   * \param level Le niveau.
   * \return La taille de la grille en X.
   */
  CartCoord globalNbFacesXCartesianView(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la taille de la vue "grille cartésienne"
   *        contenant les faces.
   *
   * Un exemple de cette vue est disponible dans la documentation de \a globalNbFacesXCartesianView.
   *
   * \param level Le niveau.
   * \return La taille de la grille en Y.
   */
  CartCoord globalNbFacesYCartesianView(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la taille de la vue "grille cartésienne"
   *        contenant les faces.
   *
   * Un exemple de cette vue est disponible dans la documentation de \a globalNbFacesXCartesianView.
   *
   * \param level Le niveau.
   * \return La taille de la grille en Z.
   */
  CartCoord globalNbFacesZCartesianView(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de mailles total dans un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de mailles dans le niveau.
   */
  Int64 nbCellInLevel(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de noeuds total dans un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de noeuds dans le niveau.
   */
  Int64 nbNodeInLevel(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de faces total dans un niveau.
   *
   * \param level Le niveau.
   * \return Le nombre de faces dans le niveau.
   */
  Int64 nbFaceInLevel(Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le pattern de raffinement utilisé dans chaque maille.
   * Par exemple, si le pattern vaut 2, chaque maille parente aura 2*2 mailles filles (2*2*2 en 3D).
   *
   * \return Le pattern de raffinement.
   */
  Int32 pattern() const;

  /*!
   * \brief Méthode permettant de récupérer le niveau d'une maille avec son uid.
   *
   * \param uid L'uniqueId de la maille.
   * \return Le niveau de la maille.
   */
  Int32 cellLevel(Int64 uid) const;

  /*!
   * \brief Méthode permettant de récupérer le niveau d'un noeud avec son uid.
   *
   * \param uid L'uniqueId du noeud.
   * \return Le niveau du noeud.
   */
  Int32 nodeLevel(Int64 uid) const;

  /*!
   * \brief Méthode permettant de récupérer le niveau d'une face avec son uid.
   *
   * \param uid L'uniqueId de la face.
   * \return Le niveau de la face.
   */
  Int32 faceLevel(Int64 uid) const;

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
  CartCoord offsetLevelToLevel(CartCoord coord, Int32 level_from, Int32 level_to) const;

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
  CartCoord faceOffsetLevelToLevel(CartCoord coord, Int32 level_from, Int32 level_to) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'une maille grâce à son uniqueId.
   *
   * \param uid L'uniqueId de la maille.
   * \param level Le niveau de la maille.
   * \return La position en X de la maille.
   */
  CartCoord cellUniqueIdToCoordX(Int64 uid, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'une maille.
   *
   * \param cell La maille.
   * \return La position en X de la maille.
   */
  CartCoord cellUniqueIdToCoordX(Cell cell) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'une maille grâce à son uniqueId.
   *
   * \param uid L'uniqueId de la maille.
   * \param level Le niveau de la maille.
   * \return La position en Y de la maille.
   */
  CartCoord cellUniqueIdToCoordY(Int64 uid, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'une maille.
   *
   * \param cell La maille.
   * \return La position en Y de la maille.
   */
  CartCoord cellUniqueIdToCoordY(Cell cell) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'une maille grâce à son uniqueId.
   *
   * \param uid L'uniqueId de la maille.
   * \param level Le niveau de la maille.
   * \return La position en Z de la maille.
   */
  CartCoord cellUniqueIdToCoordZ(Int64 uid, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'une maille.
   *
   * \param cell La maille.
   * \return La position en Z de la maille.
   */
  CartCoord cellUniqueIdToCoordZ(Cell cell) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'un noeud grâce à son uniqueId.
   *
   * \param uid L'uniqueId du noeud.
   * \param level Le niveau du noeud.
   * \return La position en X du noeud.
   */
  CartCoord nodeUniqueIdToCoordX(Int64 uid, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'un noeud.
   *
   * \param node Le noeud.
   * \return La position en X du noeud.
   */
  CartCoord nodeUniqueIdToCoordX(Node node) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'un noeud grâce à son uniqueId.
   *
   * \param uid L'uniqueId du noeud.
   * \param level Le niveau du noeud.
   * \return La position en Y du noeud.
   */
  CartCoord nodeUniqueIdToCoordY(Int64 uid, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'un noeud.
   *
   * \param node Le noeud.
   * \return La position en Y du noeud.
   */
  CartCoord nodeUniqueIdToCoordY(Node node) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'un noeud grâce à son uniqueId.
   *
   * \param uid L'uniqueId du noeud.
   * \param level Le niveau du noeud.
   * \return La position en Z du noeud.
   */
  CartCoord nodeUniqueIdToCoordZ(Int64 uid, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'un noeud.
   *
   * \param node Le noeud.
   * \return La position en Z du noeud.
   */
  CartCoord nodeUniqueIdToCoordZ(Node node) const;

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
  CartCoord faceUniqueIdToCoordX(Int64 uid, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en X d'une face.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param face La face.
   * \return La position en X de la face.
   */
  CartCoord faceUniqueIdToCoordX(Face face) const;

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
  CartCoord faceUniqueIdToCoordY(Int64 uid, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Y d'une face.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param face La face.
   * \return La position en Y de la face.
   */
  CartCoord faceUniqueIdToCoordY(Face face) const;

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
  CartCoord faceUniqueIdToCoordZ(Int64 uid, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer la coordonnée en Z d'une face.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param face La face.
   * \return La position en Z de la face.
   */
  CartCoord faceUniqueIdToCoordZ(Face face) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une maille à partir de sa position et de son niveau.
   *
   * \param cell_coord La position de la maille.
   * \param level Le niveau de la maille.
   * \return L'uniqueId de la maille.
   */
  Int64 cellUniqueId(CartCoord3 cell_coord, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une maille à partir de sa position et de son niveau.
   *
   * \param cell_coord La position de la maille.
   * \param level Le niveau de la maille.
   * \return L'uniqueId de la maille.
   */
  Int64 cellUniqueId(CartCoord2 cell_coord, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'un noeud à partir de sa position et de son niveau.
   *
   * \param node_coord La position du noeud.
   * \param level Le niveau du noeud.
   * \return L'uniqueId du noeud.
   */
  Int64 nodeUniqueId(CartCoord3 node_coord, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'un noeud à partir de sa position et de son niveau.
   *
   * \param node_coord La position du noeud.
   * \param level Le niveau du noeud.
   * \return L'uniqueId du noeud.
   */
  Int64 nodeUniqueId(CartCoord2 node_coord, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une face à partir de sa position et de son niveau.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param face_coord La position de la face.
   * \param level Le niveau de la face.
   * \return L'uniqueId de la face.
   */
  Int64 faceUniqueId(CartCoord3 face_coord, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une face à partir de sa position et de son niveau.
   *
   * Attention, les coordonnées utilisées ici sont les coordonnées des faces en "vue cartésienne"
   * (voir \a globalNbFacesXCartesianView ).
   *
   * \param face_coord La position de la face.
   * \param level Le niveau de la face.
   * \return L'uniqueId de la face.
   */
  Int64 faceUniqueId(CartCoord2 face_coord, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de noeuds dans une maille.
   *
   * \return Le nombre de noeuds d'une maille.
   */
  Int32 nbNodeByCell() const;

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
   * \param cell_coord La position de la maille.
   * \param level Le niveau de la maille (et donc des noeuds).
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbNodeByCell().
   */
  void cellNodeUniqueIds(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) const;

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
   * \param cell_coord La position de la maille.
   * \param level Le niveau de la maille (et donc des noeuds).
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbNodeByCell().
   */
  void cellNodeUniqueIds(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) const;

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
   * \param cell_uid L'uniqueId de la maille.
   * \param level Le niveau de la maille (et donc des noeuds).
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbNodeByCell().
   */
  void cellNodeUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des noeuds d'une maille.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des noeuds
   * d'une maille d'Arcane.
   *      3--2
   * ^y   |  |
   * |    0--1
   * ->x
   *
   * \param cell La maille.
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbNodeByCell().
   */
  void cellNodeUniqueIds(Cell cell, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de faces dans une maille.
   *
   * \return Le nombre de faces d'une maille.
   */
  Int32 nbFaceByCell() const;

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
   * \param cell_coord La position de la maille.
   * \param level Le niveau de la maille (et donc des faces).
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbFaceByCell().
   */
  void cellFaceUniqueIds(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) const;

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
   * \param cell_coord La position de la maille.
   * \param level Le niveau de la maille (et donc des faces).
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbFaceByCell().
   */
  void cellFaceUniqueIds(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) const;

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
   * \param cell_uid L'uniqueId de la maille.
   * \param level Le niveau de la maille (et donc des faces).
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbFaceByCell().
   */
  void cellFaceUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des faces d'une maille.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des faces
   * d'une maille d'Arcane.
   *      -2-
   * ^y   3 1
   * |    -0-
   * ->x
   *
   * \param cell La maille.
   * \param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à nbFaceByCell().
   */
  void cellFaceUniqueIds(Cell cell, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des mailles autour d'une maille.
   *
   * S'il n'y a pas de maille à un endroit autour (si on est au bord du maillage par exemple),
   * on met un uniqueId = -1.
   *
   * La vue passée en paramètre doit faire une taille de 27.
   *
   * \param cell_coord La position de la maille.
   * \param level Le niveau de la maille au centre.
   * \param uid [OUT] Les uniqueIds des mailles autour.
   */
  void cellUniqueIdsAroundCell(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des mailles autour d'une maille.
   *
   * S'il n'y a pas de maille à un endroit autour (si on est au bord du maillage par exemple),
   * on met un uniqueId = -1.
   *
   * La vue passée en paramètre doit faire une taille de 9.
   *
   * \param cell_coord La position de la maille.
   * \param level Le niveau de la maille au centre.
   * \param uid [OUT] Les uniqueIds des mailles autour.
   */
  void cellUniqueIdsAroundCell(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des mailles autour de la maille passée
   * en paramètre.
   *
   * S'il n'y a pas de maille à un endroit autour (si on est au bord du maillage par exemple),
   * on met un uniqueId = -1.
   *
   * La vue passée en paramètre doit faire une taille de 9 en 2D et de 27 en 3D.
   *
   * \param cell_uid L'uniqueId de la maille au centre.
   * \param level Le niveau de la maille au centre.
   * \param uid [OUT] Les uniqueIds des mailles autour.
   */
  void cellUniqueIdsAroundCell(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des mailles autour de la maille passée
   * en paramètre.
   *
   * S'il n'y a pas de maille à un endroit autour (si on est au bord du maillage par exemple),
   * on met un uniqueId = -1.
   *
   * La vue passée en paramètre doit faire une taille de 9 en 2D et de 27 en 3D.
   *
   * \param cell La maille au centre.
   * \param uid [OUT] Les uniqueIds des mailles autour.
   */
  void cellUniqueIdsAroundCell(Cell cell, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des mailles autour d'un noeud.
   *
   * S'il n'y a pas de maille à un endroit autour (si on est au bord du maillage par exemple),
   * on met un uniqueId = -1.
   *
   * La vue passée en paramètre doit faire une taille de 8.
   *
   * \param node_coord La position du noeud.
   * \param level Le niveau du noeud.
   * \param uid [OUT] Les uniqueIds des mailles autour.
   */
  void cellUniqueIdsAroundNode(CartCoord3 node_coord, Int32 level, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des mailles autour d'un noeud.
   *
   * S'il n'y a pas de maille à un endroit autour (si on est au bord du maillage par exemple),
   * on met un uniqueId = -1.
   *
   * La vue passée en paramètre doit faire une taille de 4.
   *
   * \param node_coord La position du noeud.
   * \param level Le niveau du noeud.
   * \param uid [OUT] Les uniqueIds des mailles autour.
   */
  void cellUniqueIdsAroundNode(CartCoord2 node_coord, Int32 level, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des mailles autour du noeud passée
   * en paramètre.
   *
   * S'il n'y a pas de maille à un endroit autour (si on est au bord du maillage par exemple),
   * on met un uniqueId = -1.
   *
   * La vue passée en paramètre doit faire une taille de 4 en 2D ou de 8 en 3D.
   *
   * \param node_uid L'uniqueId du noeud.
   * \param level Le niveau du noeud.
   * \param uid [OUT] Les uniqueIds des mailles autour.
   */
  void cellUniqueIdsAroundNode(Int64 node_uid, Int32 level, ArrayView<Int64> uid) const;

  /*!
   * \brief Méthode permettant de récupérer les uniqueIds des mailles autour du noeud passée
   * en paramètre.
   *
   * S'il n'y a pas de maille à un endroit autour (si on est au bord du maillage par exemple),
   * on met un uniqueId = -1.
   *
   * La vue passée en paramètre doit faire une taille de 4 en 2D ou de 8 en 3D.
   *
   * \param node Le noeud.
   * \param uid [OUT] Les uniqueIds des mailles autour.
   */
  void cellUniqueIdsAroundNode(Node node, ArrayView<Int64> uid) const;

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
  Int64 parentCellUniqueIdOfCell(Int64 uid, Int32 level, bool do_fatal = true) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId du parent d'une maille.
   *
   * Si \a do_fatal est vrai, une erreur fatale est générée si le parent n'existe
   * pas, sinon l'uniqueId retourné a pour valeur NULL_ITEM_UNIQUE_ID.
   *
   * \param cell La maille enfant.
   * \return L'uniqueId de la maille parent de la maille passé en paramètre.
   */
  Int64 parentCellUniqueIdOfCell(Cell cell, bool do_fatal = true) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une maille enfant d'une maille parent
   * à partir de la position de la maille enfant dans la maille parent.
   *
   * \param cell La maille parent.
   * \param child_coord_in_parent La position de l'enfant dans la maille parent.
   * \return L'uniqueId de la maille enfant demandée.
   */
  Int64 childCellUniqueIdOfCell(Cell cell, CartCoord3 child_coord_in_parent) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une maille enfant d'une maille parent
   * à partir de la position de la maille enfant dans la maille parent.
   *
   * \param cell La maille parent.
   * \param child_coord_in_parent La position de l'enfant dans la maille parent.
   * \return L'uniqueId de la maille enfant demandée.
   */
  Int64 childCellUniqueIdOfCell(Cell cell, CartCoord2 child_coord_in_parent) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une maille enfant d'une maille parent
   * à partir de l'index de la maille enfant dans la maille parent.
   *
   * \param cell La maille parent.
   * \param child_index_in_parent L'index de l'enfant dans la maille parent.
   * \return L'uniqueId de la maille enfant demandée.
   */
  Int64 childCellUniqueIdOfCell(Cell cell, Int32 child_index_in_parent) const;

  /*!
   * \brief Méthode permettant de récupérer une maille enfant d'une maille parent
   * à partir de la position de la maille enfant dans la maille parent.
   *
   * \param cell La maille parent.
   * \param child_coord_in_parent La position de l'enfant dans la maille parent.
   * \return La maille enfant demandée.
   */
  Cell childCellOfCell(Cell cell, CartCoord3 child_coord_in_parent) const;

  /*!
   * \brief Méthode permettant de récupérer une maille enfant d'une maille parent
   * à partir de la position de la maille enfant dans la maille parent.
   *
   * \param cell La maille parent.
   * \param child_coord_in_parent La position de l'enfant dans la maille parent.
   * \return La maille enfant demandée.
   */
  Cell childCellOfCell(Cell cell, CartCoord2 child_coord_in_parent) const;

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
  Int64 parentNodeUniqueIdOfNode(Int64 uid, Int32 level, bool do_fatal = true) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId du parent d'un noeud.
   *
   * Si \a do_fatal est vrai, une erreur fatale est générée si le parent n'existe
   * pas, sinon l'uniqueId retourné a pour valeur NULL_ITEM_UNIQUE_ID.
   *
   * \param node Le noeud enfant.
   * \return L'uniqueId du noeud parent du noeud passé en paramètre.
   */
  Int64 parentNodeUniqueIdOfNode(Node node, bool do_fatal = true) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'un noeud enfant d'un noeud parent.
   *
   * \param uid L'uniqueId du noeud enfant.
   * \param level Le niveau du noeud enfant.
   * \return L'uniqueId du noeud enfant demandée.
   */
  Int64 childNodeUniqueIdOfNode(Int64 uid, Int32 level) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'un noeud enfant d'un noeud parent.
   *
   * \param node Le noeud parent.
   * \return L'uniqueId du noeud enfant demandée.
   */
  Int64 childNodeUniqueIdOfNode(Node node) const;

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
  Int64 parentFaceUniqueIdOfFace(Int64 uid, Int32 level, bool do_fatal = true) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId du parent d'une face.
   *
   * Si \a do_fatal est vrai, une erreur fatale est générée si le parent n'existe
   * pas, sinon l'uniqueId retourné a pour valeur NULL_ITEM_UNIQUE_ID.
   *
   * \param face La face enfant.
   * \return L'uniqueId de la face parent de la face passé en paramètre.
   */
  Int64 parentFaceUniqueIdOfFace(Face face, bool do_fatal = true) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une face enfant d'une face parent
   * à partir de l'index de la face enfant dans la face parent.
   *
   * \param uid L'uniqueId de la face parent.
   * \param level Le niveau de la face parent.
   * \param child_index_in_parent L'index de l'enfant dans la face parent.
   * \return L'uniqueId de la face enfant demandée.
   */
  Int64 childFaceUniqueIdOfFace(Int64 uid, Int32 level, Int32 child_index_in_parent) const;

  /*!
   * \brief Méthode permettant de récupérer l'uniqueId d'une face enfant d'une face parent
   * à partir de l'index de la face enfant dans la face parent.
   *
   * \param face La face parent.
   * \param child_index_in_parent L'index de l'enfant dans la face parent.
   * \return L'uniqueId de la face enfant demandée.
   */
  Int64 childFaceUniqueIdOfFace(Face face, Int32 child_index_in_parent) const;

 public:

  ICartesianMeshNumberingMngInternal* _internalApi() const;

 private:

  Ref<ICartesianMeshNumberingMngInternal> m_internal_api;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHNUMBERINGMNG_H
