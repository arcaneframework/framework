// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeInfo.h                                              (C) 2000-2025 */
/*                                                                           */
/* Informations sur un type d'entité du maillage.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMTYPEINFO_H
#define ARCANE_CORE_ITEMTYPEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/ItemTypeId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Infos sur un type d'entité du maillage.
 *
 * Une instance de cette classe décrit un type d'entité de maillage, par
 * exemple une maille hexédrique, une maille quadrangulaire.
 *
 * \sa ItemTypeMng
 *
 * Il ne doit exister qu'une instance par type d'entité. La création d'un
 * type se fait par la classe dérivée ItemTypeInfoBuilder. Les types doivent
 * être créé avant toute création de maillage (i.e durant
 * l'initialisation de l'architecture).
 */
class ItemTypeInfo
{
 public:

  /*!
   * \brief Informations locales sur une face d'une maille.
   */
  class LocalFace
  {
   public:

    LocalFace(Integer* index)
    : m_index(index)
    {}

   public:

    //! Type de l'entité face
    Integer typeId() const { return m_index[0]; }
    //! Nombre de noeuds de la face
    Integer nbNode() const { return m_index[1]; }
    //! Indice locale dans la maille du i-ème noeud de la face
    Integer node(Integer i) const { return m_index[2 + i]; }
    //! Nombre d'arête de la face
    Integer nbEdge() const { return m_index[2 + nbNode()]; }
    //! Arête de la face
    Integer edge(Integer i) const { return m_index[3 + nbNode() + i]; }

   private:

    Integer* m_index; //!< Indices dans le tampon des infos de la face
  };

  /*!
   * \brief Informations locales sur une arête d'une maille.
   *
   * \warning Pour être initialisée comme un tableau, cette classe doit être
   * un POD et ne pas avoir de constructeur.
   */
  class LocalEdge
  {
   public:

    LocalEdge(Integer* index)
    : m_index(index)
    {}

   public:

    //! Indice local à la maille du sommet origine de l'arête
    Integer beginNode() const { return m_index[0]; }
    //! Indice local à la maille du sommet extrémité de l'arête
    Integer endNode() const { return m_index[1]; }
    //! Indice local à la maille de la face à gauche de l'arête
    Integer leftFace() const { return m_index[2]; }
    //! Indice local à la maille du la face à droite de l'arête
    Integer rightFace() const { return m_index[3]; }

   private:

    Integer* m_index; //!< Indices dans le tampon des infos de la face
  };

 protected:

  //! Constructeur par défaut
  ItemTypeInfo() = default;

 public:

  //! Numéro du type
  Int16 typeId() const { return m_type_id.typeId(); }
  //! Numéro du type
  ItemTypeId itemTypeId() const { return m_type_id; }
  //! Nombre de noeuds de l'entité
  Integer nbLocalNode() const { return m_nb_node; }
  //! Nombre de faces de l'entité
  Integer nbLocalFace() const { return m_nb_face; }
  //! Nombre d'arêtes de l'entité
  Integer nbLocalEdge() const { return m_nb_edge; }
  //! Nom du type
  String typeName() const { return m_type_name; }
  //! Dimension de l'élément (<0 si inconnu)
  Int16 dimension() const { return m_dimension; }
  //! Indique si le type est valide pour créér une maille (Cell)
  bool isValidForCell() const { return m_is_valid_for_cell; }
  //! Ordre du type
  Int32 order() const { return m_order; }
  //! Type de l'élément linéaire correspondant
  Int16 linearTypeId() const { return m_linear_type_id.typeId(); }
  //! Type de l'élément linéaire correspondant
  ItemTypeId linearItemTypeId() const { return m_linear_type_id; }

 public:

  //! Connectivité locale de la \a i-ème arête de la maille
  LocalEdge localEdge(Integer id) const
  {
    Array<Integer>& buf = m_mng->m_ids_buffer;
    Integer fi = buf[m_first_item_index + id];
    return LocalEdge(&buf[fi]);
  }

  //! Connectivité locale de la \a i-ème face de la maille
  LocalFace localFace(Integer id) const
  {
    Array<Integer>& buf = m_mng->m_ids_buffer;
    Integer fi = buf[m_first_item_index + m_nb_edge + id];
    return LocalFace(&buf[fi]);
  }

 protected:

  ItemTypeMng* m_mng = nullptr;
  ItemTypeId m_type_id{ IT_NullType };
  ItemTypeId m_linear_type_id{ IT_NullType };
  //! Dimension (-1) si pas initialisé.
  Int16 m_dimension = (-1);
  //! Indique si le type est valide pour une maille.
  bool m_is_valid_for_cell = true;
  Integer m_nb_node = 0;
  Integer m_nb_edge = 0;
  Integer m_nb_face = 0;
  Int32 m_order = 1;
  //! Indice de ce type dans la liste des index de \a m_mng.
  Integer m_first_item_index = 0;
  String m_type_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

