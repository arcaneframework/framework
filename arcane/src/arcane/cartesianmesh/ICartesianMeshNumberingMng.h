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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshNumberingMng
{
 public:
  ~ICartesianMeshNumberingMng() = default;

 public:

  virtual void prepareLevel(Int32 level) =0;

  virtual void updateFirstLevel() = 0;

  /*!
   * @brief Méthode permettant de récupérer le premier unique id utilisé par les mailles d'un niveau.
   * L'appel de cette méthode avec level et level+1 permet de récupérer l'intervalle des uniqueids
   * d'un niveau.
   *
   * @param level Le niveau.
   * @return Le premier uid des mailles du niveau.
   */
  virtual Int64 getFirstCellUidLevel(Integer level) =0;

  /*!
   * @brief Méthode permettant de récupérer le premier unique id utilisé par les noeuds d'un niveau.
   * L'appel de cette méthode avec level et level+1 permet de récupérer l'intervalle des uniqueids
   * d'un niveau.
   *
   * @param level Le niveau.
   * @return Le premier uid des noeuds du niveau.
   */
  virtual Int64 getFirstNodeUidLevel(Integer level) =0;

  /*!
   * @brief Méthode permettant de récupérer le premier unique id utilisé par les faces d'un niveau.
   * L'appel de cette méthode avec level et level+1 permet de récupérer l'intervalle des uniqueids
   * d'un niveau.
   *
   * @param level Le niveau.
   * @return Le premier uid des faces du niveau.
   */
  virtual Int64 getFirstFaceUidLevel(Integer level) =0;

  /*!
   * @brief Méthode permettant de récupérer le nombre de mailles globale en X d'un niveau.
   *
   * @param level Le niveau.
   * @return Le nombre de mailles en X.
   */
  virtual Int64 getGlobalNbCellsX(Integer level) const =0;

  /*!
   * @brief Méthode permettant de récupérer le nombre de mailles globale en Y d'un niveau.
   *
   * @param level Le niveau.
   * @return Le nombre de mailles en Y.
   */
  virtual Int64 getGlobalNbCellsY(Integer level) const =0;

  /*!
   * @brief Méthode permettant de récupérer le nombre de mailles globale en Z d'un niveau.
   *
   * @param level Le niveau.
   * @return Le nombre de mailles en Z.
   */
  virtual Int64 getGlobalNbCellsZ(Integer level) const =0;


  /*!
   * @brief Méthode permettant de récupérer le pattern de raffinement utilisé dans chaque maille.
   * Par exemple, si le pattern vaut 2, chaque maille parente aura 2*2 mailles filles (2*2*2 en 3D).
   *
   * @return Le pattern de raffinement.
   */
  virtual Integer getPattern() const =0;

  /*!
   * @brief Méthode permettant d'obtenir la position de la première maille fille à partir de la position
   * de la maille parente.
   *
   * Exemple : si l'on a un maillage 2D de 2*2 mailles et un pattern de raffinement de 2,
   * on sait que la grille de niveau 1 (pour les patchs de niveau 1) sera de 4*4 mailles.
   * La première maille fille de la maille parente (Xp=1,Yp=0) aura la position Xf=Xp*Pattern=2 (idem pour Y).
   *
   * @param coord La position X ou Y ou Z de la maille parente.
   * @param level_from Le niveau parent.
   * @param level_to Le niveau enfant.
   * @return La position de la première fille de la maille parente.
   */
  virtual Int64 getOffsetLevelToLevel(Int64 coord, Integer level_from, Integer level_to) const =0;


  /*!
   * @brief Méthode permettant de récupérer la coordonnée en X d'une maille grâce à son uniqueId.
   *
   * @param uid L'uniqueId de la maille.
   * @param level Le niveau de la maille.
   * @return La position en X de la maille.
   */
  virtual Int64 uidToCoordX(Int64 uid, Integer level) =0;

  /*!
   * @brief Méthode permettant de récupérer la coordonnée en X d'une maille.
   *
   * @param cell La maille.
   * @return La position en X de la maille.
   */
  virtual Int64 uidToCoordX(Cell cell) =0;


  /*!
   * @brief Méthode permettant de récupérer la coordonnée en Y d'une maille grâce à son uniqueId.
   *
   * @param uid L'uniqueId de la maille.
   * @param level Le niveau de la maille.
   * @return La position en Y de la maille.
   */
  virtual Int64 uidToCoordY(Int64 uid, Integer level) =0;

  /*!
   * @brief Méthode permettant de récupérer la coordonnée en Y d'une maille.
   *
   * @param cell La maille.
   * @return La position en Y de la maille.
   */
  virtual Int64 uidToCoordY(Cell cell) =0;


  /*!
   * @brief Méthode permettant de récupérer la coordonnée en Z d'une maille grâce à son uniqueId.
   *
   * @param uid L'uniqueId de la maille.
   * @param level Le niveau de la maille.
   * @return La position en Z de la maille.
   */
  virtual Int64 uidToCoordZ(Int64 uid, Integer level) =0;

  /*!
   * @brief Méthode permettant de récupérer la coordonnée en Z d'une maille.
   *
   * @param cell La maille.
   * @return La position en Z de la maille.
   */
  virtual Int64 uidToCoordZ(Cell cell) =0;


  /*!
   * @brief Méthode permettant de récupérer l'uniqueId d'une maille à partir de sa position et de son niveau.
   *
   * @param level Le niveau de la maille.
   * @param cell_coord_i La position X de la maille.
   * @param cell_coord_j La position Y de la maille.
   * @param cell_coord_k La position Z de la maille.
   * @return L'uniqueId de la maille.
   */
  virtual Int64 getCellUid(Integer level, Int64 cell_coord_i, Int64 cell_coord_j, Int64 cell_coord_k) =0;

  /*!
   * @brief Méthode permettant de récupérer l'uniqueId d'une maille à partir de sa position et de son niveau.
   *
   * @param level Le niveau de la maille.
   * @param cell_coord_i La position X de la maille.
   * @param cell_coord_j La position Y de la maille.
   * @return L'uniqueId de la maille.
   */
  virtual Int64 getCellUid(Integer level, Int64 cell_coord_i, Int64 cell_coord_j) =0;


  /*!
   * @brief Méthode permettant de récupérer le nombre de noeuds dans une maille.
   *
   * @return Le nombre de noeuds d'une maille.
   */
  virtual Integer getNbNode() =0;

  /*!
   * @brief Méthode permettant de récupérer les uniqueIds des noeuds d'une maille à partir de
   * ces coordonnées.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des noeuds
   * d'une maille d'Arcane.
   *      3--2
   * ^y   |  |
   * |    0--1
   * ->x
   *
   * @param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à getNbNode().
   * @param level Le niveau de la maille (et donc des noeuds).
   * @param cell_coord_i La position X de la maille.
   * @param cell_coord_j La position Y de la maille.
   * @param cell_coord_k La position Z de la maille.
   */
  virtual void getNodeUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j, Int64 cell_coord_k) =0;

  /*!
   * @brief Méthode permettant de récupérer les uniqueIds des noeuds d'une maille à partir de
   * ces coordonnées.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des noeuds
   * d'une maille d'Arcane.
   *      3--2
   * ^y   |  |
   * |    0--1
   * ->x
   *
   * @param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à getNbNode().
   * @param level Le niveau de la maille (et donc des noeuds).
   * @param cell_coord_i La position X de la maille.
   * @param cell_coord_j La position Y de la maille.
   */
  virtual void getNodeUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j) =0;

  virtual void getNodeUids(ArrayView<Int64> uid, Integer level, Int64 cell_uid) =0;


  /*!
   * @brief Méthode permettant de récupérer le nombre de faces dans une maille.
   *
   * @return Le nombre de faces d'une maille.
   */
  virtual Integer getNbFace() =0;

  /*!
   * @brief Méthode permettant de récupérer les uniqueIds des faces d'une maille à partir de
   * ces coordonnées.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des faces
   * d'une maille d'Arcane.
   *      -2-
   * ^y   3 1
   * |    -0-
   * ->x
   *
   * @param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à getNbFace().
   * @param level Le niveau de la maille (et donc des faces).
   * @param cell_coord_i La position X de la maille.
   * @param cell_coord_j La position Y de la maille.
   * @param cell_coord_k La position Z de la maille.
   */
  virtual void getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j, Int64 cell_coord_k) =0;

  /*!
   * @brief Méthode permettant de récupérer les uniqueIds des faces d'une maille à partir de
   * ces coordonnées.
   *
   * L'ordre dans lequel les uniqueIds sont placés correspond à l'ordre d'énumération des faces
   * d'une maille d'Arcane.
   *      -2-
   * ^y   3 1
   * |    -0-
   * ->x
   *
   * @param uid [OUT] Les uniqueIds de la maille. La taille de l'ArrayView doit être égal à getNbFace().
   * @param level Le niveau de la maille (et donc des faces).
   * @param cell_coord_i La position X de la maille.
   * @param cell_coord_j La position Y de la maille.
   */
  virtual void getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_coord_i, Int64 cell_coord_j) =0;

  virtual void getFaceUids(ArrayView<Int64> uid, Integer level, Int64 cell_uid) =0;

  /*!
   * @brief Méthode permettant de récupérer les uniqueIds des mailles autour de la maille passée
   * en paramètre.
   *
   * La vue passée en paramètre doit faire une taille de 9 en 2D et de 27 en 3D.
   *
   * @param uid [OUT] Les uniqueIds des mailles autour.
   * @param cell_uid L'uniqueId de la maille au centre.
   * @param level Le niveau de la maille au centre.
   */
  virtual void getCellUidsAround(ArrayView<Int64> uid, Int64 cell_uid, Int32 level) = 0;

  /*!
   * @brief Méthode permettant de récupérer les uniqueIds des mailles autour de la maille passée
   * en paramètre.
   *
   * La vue passée en paramètre doit faire une taille de 9 en 2D et de 27 en 3D.
   *
   * @param uid [OUT] Les uniqueIds des mailles autour.
   * @param cell La maille au centre.
   */
  virtual void getCellUidsAround(ArrayView<Int64> uid, Cell cell) =0;


  /*!
   * @brief Méthode permettant de définir les coordonnées spatiales des nodes d'une maille enfant.
   * Cette méthode doit être appelée après l'appel à endUpdate().
   *
   * @param child_cell La maille enfant.
   */
  virtual void setChildNodeCoordinates(Cell child_cell) = 0;

  /*!
   * @brief Méthode permettant de définir les coordonnées spatiales des nodes d'une maille parent.
   * Cette méthode doit être appelée après l'appel à endUpdate().
   *
   * @param parent_cell La maille parent.
   */
  virtual void setParentNodeCoordinates(Cell parent_cell) = 0;

  /*!
   * \brief TODO
   * \param cell
   * \return
   */
  virtual Int64 getParentCellUidOfCell(Cell cell) = 0;

  /*!
   * \brief TODO
   * \param cell
   * \param child_index
   * \return
   */
  virtual Int64 getChildCellUidOfCell(Cell cell, Int64 child_coord_x_in_parent, Int64 child_coord_y_in_parent) = 0;

  /*!
   * \brief TODO
   * \param cell
   * \param child_index
   * \return
   */
  virtual Int64 getChildCellUidOfCell(Cell cell, Int64 child_coord_x_in_parent, Int64 child_coord_y_in_parent, Int64 child_coord_z_in_parent) = 0;

  virtual Int64 getChildCellUidOfCell(Cell cell, Int64 child_index_in_parent) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_ICARTESIANMESHNUMBERINGMNG_H
