﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeInfoBuilder.h                                       (C) 2000-2022 */
/*                                                                           */
/* Construction d'un type d'entité du maillage.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMTYPEINFOBUILDER_H
#define ARCANE_ITEMTYPEINFOBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypeInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Construction des infos d'un type d'entité du maillage.
 
 Pour des raisons de performances, on essaie de stocker ces informations
 de manière contigue en mémoire car elles seront accédées très souvent.
 Pour cela, on utilise un Buffer dans ItemTypeMng.

 La méthode setInfos() permet d'indiquer le type et le nombre de noeuds, d'arêtes
 et de faces du type. Il faut ensuite appeler les méthodes addEdge() et addFace...()

 Une fois le type complètement défini, il ne doit plus être modifié.

 Pour un numéro de type donné, il n'existe qu'une instance de ItemTypeInfo et cette
 instance reste valide tant que le gestionnaire de type (ItemTypeMng) n'est pas détruit.
 Par conséquent, il est possible de stocker le pointeur sur l'instance et
 de comparer deux types uniquement en comparant les pointeurs
*/
class ItemTypeInfoBuilder
: public ItemTypeInfo
{
 public:

  //! Constructeur par défaut
  ItemTypeInfoBuilder() : ItemTypeInfo() {}

 public:

  // TODO: Rendre obsolète
  void setInfos(ItemTypeMng* mng,
                Integer type_id, String type_name,
                Integer nb_node, Integer nb_edge, Integer nb_face);

  void setInfos(ItemTypeMng* mng,
                ItemTypeId type_id, String type_name,
                Integer nb_node, Integer nb_edge, Integer nb_face);

  /*!
   * \brief Ajoute une arête à la liste des arêtes
   *
   * \a n0 noeud origine
   * \a n1 noeud extremité
   * \a f_left numéro local de la face à gauche
   * \a f_right numéro local de la face à droite
   */
  void addEdge(Integer edge_index,Integer n0,Integer n1,Integer f_left,Integer f_right);

  //! Ajoute un sommet à la liste des faces (pour les elements 1D)
  void addFaceVertex(Integer face_index,Integer n0);

  //! Ajoute une ligne à la liste des faces (pour les elements 2D)
  void addFaceLine(Integer face_index,Integer n0,Integer n1);

  //! Ajoute une ligne quadratique à la liste des faces (pour les elements 2D)
  void addFaceLine3(Integer face_index,Integer n0,Integer n1,Integer n2);

  //! Ajoute un triangle à la liste des faces
  void addFaceTriangle(Integer face_index,Integer n0,Integer n1,Integer n2);

  //! Ajoute un triangle quadratique à la liste des faces
  void addFaceTriangle6(Integer face_index,Integer n0,Integer n1,Integer n2,
                        Integer n3, Integer n4, Integer n5);

  //! Ajoute un quadrilatère à la liste des faces
  void addFaceQuad(Integer face_index,Integer n0,Integer n1,Integer n2,Integer n3);

  //! Ajoute un quadrilatère quadratique à la liste des faces
  void addFaceQuad8(Integer face_index,Integer n0,Integer n1,Integer n2,Integer n3,
                    Integer n4,Integer n5,Integer n6,Integer n7);

  //! Ajoute un pentagone à la liste des faces
  void addFacePentagon(Integer face_index,Integer n0,Integer n1,Integer n2,Integer n3,Integer n4);

  //! Ajoute un hexagone à la liste des faces
  void addFaceHexagon(Integer face_index,Integer n0,Integer n1,Integer n2,Integer n3,
                      Integer n4,Integer n5);

  //! Ajoute un heptagone à la liste des faces
  void addFaceHeptagon(Integer face_index,Integer n0,Integer n1,Integer n2,Integer n3,
                       Integer n4,Integer n5, Integer n6);

  //! Ajoute un heptagone à la liste des faces
  void addFaceOctogon(Integer face_index,Integer n0,Integer n1,Integer n2,Integer n3,
                      Integer n4,Integer n5, Integer n6, Integer n7);

  //! Ajoute une face générique à la liste des faces
  void addFaceGeneric(Integer face_index,Integer type_id,ConstArrayView<Integer> n);

  //! Calcule les relations face->arêtes
  void computeFaceEdgeInfos();

 private:
  void _setNbEdgeAndFace(Integer nb_edge,Integer nb_face);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

