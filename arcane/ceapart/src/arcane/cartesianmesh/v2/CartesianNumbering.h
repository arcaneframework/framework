// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianNumbering.h                                        (C) 2000-2021 */
/*                                                                           */
/* Gestion d'une numérotation cartésienne sur une grille d'entités.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANNUMBERING_H
#define ARCANE_CARTESIANMESH_CARTESIANNUMBERING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/cartesianmesh/v2/CartesianTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::CartesianMesh::V2
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief
 * Gestion d'une numerotation cartesienne sur une grille d'items 
 * d'une dimension au plus 3
 * Permet le passage d'un triplet (i,j,k) à un numéro id
 * Les ids vont de firstId() à firstId()+nbItem()
 *
 * Ex pour une numérotation 5x3 en 2D avec firstId() = 0 :
 *
 *       ids:
 *  2        10 11 12 13 14
 *  1        5  6  7  8  9
 *  0        0  1  2  3  4
 *  /\
 *  j  i-->  0  1  2  3  4
 */
template <typename IdType>
class CartesianNumbering
{
 public:
  //! Type pour les triplets cartésiens (i,j,k) et les triplets des dimensions (ni,nj,nk)
  using IdType3 = IdType[3];

 public:
  CartesianNumbering() {}

  ARCCORE_HOST_DEVICE CartesianNumbering(const CartesianNumbering<IdType>& rhs)
  {
    m_dimension = rhs.m_dimension;
    for (Integer d(0); d < 3; ++d) {
      m_nitems_dir[d] = rhs.m_nitems_dir[d];
      m_coef[d] = rhs.m_coef[d];
    }
    m_nitems = rhs.m_nitems;
    m_first_item_id = rhs.m_first_item_id;
  }

  void initNumbering(const IdType3& nitems_dir, Integer dimension, IdType first_item_id = 0)
  {
    m_dimension = dimension;
    m_nitems = 1;
    m_first_item_id = first_item_id;

    for (Integer d(0); d < m_dimension; ++d) {
      m_nitems_dir[d] = nitems_dir[d];
      m_nitems *= m_nitems_dir[d];
    }

    for (Integer d(m_dimension); d < 3; ++d) {
      m_nitems_dir[d] = 1;
    }

    m_coef[0] = m_coef[1] = m_coef[2] = 1;

    // Numerotation selon i, puis j, puis k
    m_coef[0] = 1;
    for (Integer d(1); d < m_dimension; ++d) {
      m_coef[d] = m_coef[d - 1] * m_nitems_dir[d - 1];
    }
  }

  //! Dimension de la grille cartésienne sur laquelle s'appuit la numérotation
  Integer dimension() const
  {
    return m_dimension;
  }

  //! Triplet du nb d'items dans chaque direction (définition de la grille)
  const IdType3& nbItem3() const
  {
    return m_nitems_dir;
  }

  //! Nb d'items dans la grille cartésienne selon la direction dir (< dimension())
  // Egal à nbItem3()[dir]
  IdType nbItemDir(Integer dir) const
  {
    return m_nitems_dir[dir];
  }

  //! Nb total d'items dans la grille cartésienne (produit du nb d'items dans chaque direction)
  IdType nbItem() const
  {
    return m_nitems;
  }

  //! Plus petit identifiant de la numérotation cartésienne de la grille
  IdType firstId() const
  {
    return m_first_item_id;
  }

  //! Offset à ajouter à id() pour obtenir l'id de l'item suivant dans la direction dir (si cet item existe)
  // Egal à delta3()[dir]
  // id(item_ijk) + deltaDir(1) = id({item_ijk[0],item_ijk[1]+1,item_ijk[0]})
  IdType deltaDir(Integer dir) const
  {
    return m_coef[dir];
  }

  //! Triplet des offsets dans toutes les directions pour passer aux items suivants dans chacune des directions
  const IdType3& delta3() const
  {
    return m_coef;
  }

  //! Passage (i,j,k) => numero
  ARCCORE_HOST_DEVICE inline IdType id(IdType i, IdType j, IdType k) const
  {
    // m_coef[0] vaut 1
    // TODO : specialisation via template
    IdType item_id;
    if (m_dimension < 3) {
      item_id = m_first_item_id + i /*m_coef[0]*/ + j * m_coef[1];
    }
    else {
      item_id = m_first_item_id + i /*m_coef[0]*/ + j * m_coef[1] + k * m_coef[2];
    }
    return item_id;
  }

  //! Passage (i,j,k) => numero
  inline IdType id(const IdType3& item_ijk) const
  {
    return id(item_ijk[0], item_ijk[1], item_ijk[2]);
  }

  //! Passage (i,j,k) => numero
  // NOTE : IdxType est un triplet de Int64 alors qu'IdType peut-être un Int32 ou un Int64
  ARCCORE_HOST_DEVICE inline IdType id(IdxType idx) const
  {
    return id(static_cast<IdType>(idx[0]), static_cast<IdType>(idx[1]), static_cast<IdType>(idx[2]));
  }

  //! Passage de numero => (i,j,k)
  void ijk(IdType item_id, IdType3& item_ijk) const
  {
    item_id -= m_first_item_id;
    // TODO : specialisation via template
    if (m_dimension < 3) {
      item_ijk[2] = 0;
      item_ijk[1] = item_id / m_coef[1];
      item_ijk[0] = item_id % m_coef[1];
    }
    else {
      item_ijk[2] = item_id / m_coef[2];
      IdType tmp = item_id % m_coef[2];
      item_ijk[1] = tmp / m_coef[1];
      item_ijk[0] = tmp % m_coef[1];
    }
  }

  //! Passage de numero => (i,j,k)
  ARCCORE_HOST_DEVICE IdxType ijk(IdType item_id) const
  {
    item_id -= m_first_item_id;
    Int64 i, j, k;
    // TODO : specialisation via template
    if (m_dimension < 3) {
      k = 0;
      j = item_id / m_coef[1];
      i = item_id % m_coef[1];
    }
    else {
      k = item_id / m_coef[2];
      IdType tmp = item_id % m_coef[2];
      j = tmp / m_coef[1];
      i = tmp % m_coef[1];
    }
    return { i, j, k };
  }

  // TODO : utiliser template spécialisée
  //! Passage numéro => i
  IdType idxDir0(IdType item_id) const
  {
    return (item_id - m_first_item_id) % m_coef[1];
  }

  //! Passage numéro => j
  IdType idxDir1(IdType item_id) const
  {
    return (m_dimension == 3 ? ((item_id - m_first_item_id) % m_coef[2] / m_coef[1]) : (item_id - m_first_item_id) / m_coef[1]);
  }

  //! Passage numéro => k
  IdType idxDir2(IdType item_id) const
  {
    return (item_id - m_first_item_id) / m_coef[2];
  }

 protected:
  Integer m_dimension = 0;

  IdType3 m_nitems_dir = { 1, 1, 1 }; // Nb d'elements par direction
  IdType m_nitems = 0; // Nb total d'elements

  IdType m_first_item_id = 0; //! item_id = m_first_item_id + numéro_cartésien(i,j,k), permet un décallage dans la numérotation

  IdType3 m_coef = { 0, 0, 0 };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::CartesianMesh::V2

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
