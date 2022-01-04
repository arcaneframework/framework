// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeInfo.h                                              (C) 2000-2006 */
/*                                                                           */
/* Informations sur un type d'entité du maillage.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMTYPEINFO_H
#define ARCANE_ITEMTYPEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/ItemTypeMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Infos sur un type d'entité du maillage.

 Une instance de cette classe décrit un type d'entité de maillage, par
 exemple une maille hexédrique, une maille quadrangulaire.

 \sa ItemTypeMng

 \internal
 
 Il ne doit exister qu'une instance par type d'entité. La création d'un
 type se fait par la classe dérivée ItemTypeInfoBuilder. Les types doivent
 être créée avant toute création de maillage (i.e durant
 l'initialisation de l'architecture).
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
    LocalFace(Integer* index) : m_index(index) {}
   public:
    //! Type de l'entité face
    Integer typeId() const { return m_index[0]; }
    //! Nombre de noeuds de la face
    Integer nbNode() const { return m_index[1]; }
    //! Indice locale dans la maille du i-ème noeud de la face
    Integer node(Integer i) const { return m_index[2+i]; }
    //! Nombre d'arête de la face
    Integer nbEdge() const { return m_index[2+nbNode()]; }
    //! Arête de la face
    Integer edge(Integer i) const { return m_index[3+nbNode()+i]; }
   private:
    Integer *m_index; //!< Indices dans le tampon des infos de la face
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
    LocalEdge(Integer* index) : m_index(index) {}
   public:
    //! Indice local à la maille du sommet origine de l'arête
    Integer beginNode() const { return m_index[0]; }
    //! Indice local à la maille du sommet extrémité de l'arête
    Integer endNode()   const { return m_index[1]; }
    //! Indice local à la maille de la face à gauche de l'arête
    Integer leftFace()  const { return m_index[2]; }
    //! Indice local à la maille du la face à droite de l'arête
    Integer rightFace() const { return m_index[3]; }
   private:
    Integer *m_index; //!< Indices dans le tampon des infos de la face
  };

 protected:

  //! Constructeur par défaut
  ItemTypeInfo();

 public:

  //! Numéro du type
  Integer typeId() const { return m_type_id; }
  //! Nombre de noeuds de l'entité
  Integer nbLocalNode() const { return m_nb_node; }
  //! Nombre de faces de l'entité
  Integer nbLocalFace() const { return m_nb_face; }
  //! Nombre d'arêtes de l'entité
  Integer nbLocalEdge() const { return m_nb_edge; }
  //! Nom du type
  const String typeName() const { return m_type_name; }

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
  ItemTypeMng* m_mng;
  Integer m_type_id;
  Integer m_nb_node;
  Integer m_nb_edge;
  Integer m_nb_face;
  Integer m_first_item_index;
  String m_type_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

