// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeInfoBuilder.h                                       (C) 2000-2025 */
/*                                                                           */
/* Construction d'un type d'entité du maillage.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMTYPEINFOBUILDER_H
#define ARCANE_CORE_ITEMTYPEINFOBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypeInfo.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Construction des infos d'un type d'entité du maillage.
 *
 * Pour des raisons de performances, on essaie de stocker ces informations
 * de manière contigue en mémoire, car elles seront accédées très souvent.
 * Pour cela, on utilise un Buffer dans ItemTypeMng.
 *
 * La méthode setInfos() permet d'indiquer le type et le nombre de noeuds, d'arêtes
 * et de faces du type. Il faut ensuite appeler les méthodes addEdge() et addFace...()
 *
 * Une fois le type complètement défini, il ne doit plus être modifié.
 *
 * Pour un numéro de type donné, il n'existe qu'une instance de ItemTypeInfo et cette
 * instance reste valide tant que le gestionnaire de type (ItemTypeMng) n'est pas détruit.
 * Par conséquent, il est possible de stocker le pointeur sur l'instance et
 * de comparer deux types uniquement en comparant les pointeurs.
 */
class ItemTypeInfoBuilder
: public ItemTypeInfo
{
 public:

  //! Dimension du type
  enum class Dimension : Int16
  {
    DimUnknown = -1,
    Dim0 = 0,
    Dim1 = 1,
    Dim2 = 2,
    Dim3 = 3
  };

 public:

  //! Constructeur par défaut
  ItemTypeInfoBuilder() = default;

 public:

  ARCANE_DEPRECATED_REASON("Y2025: Use setInfo(...,Dimension dimension, ...) instead")
  void setInfos(ItemTypeMng* mng, Int16 type_id, String type_name,
                Int32 nb_node, Int32 nb_edge, Int32 nb_face);

  ARCANE_DEPRECATED_REASON("Y2025: Use setInfo(...,Dimension dimension, ...) instead")
  void setInfos(ItemTypeMng* mng, ItemTypeId type_id, String type_name,
                Int32 nb_node, Int32 nb_edge, Int32 nb_face);

  /*!
   * \brief Positionne les informations d'un type.
   */
  ARCANE_DEPRECATED_REASON("Y2025: Use setInfo(...,Dimension dimension, ...) instead")
  void setInfos(ItemTypeMng* mng, ItemTypeId type_id, String type_name, Int16 dimension,
                Int32 nb_node, Int32 nb_edge, Int32 nb_face);

  /*!
   * \brief Positionne les informations d'un type.
   */
  void setInfos(ItemTypeMng* mng, ItemTypeId type_id, String type_name, Dimension dimension,
                Int32 nb_node, Int32 nb_edge, Int32 nb_face);

  /*!
   * \brief Positionne les informations d'un type.
   */
  void setInfos(ItemTypeMng* mng, Int16 type_id, String type_name, Dimension dimension,
                Int32 nb_node, Int32 nb_edge, Int32 nb_face);

  /*!
   * \brief Positionne l'ordre du type.
   *
   * Si pas appelé, on considère que le type est pour les entités d'ordre 1.
   * Le premier argument est l'ordre de l'entité et le deuxième l'élément d'ordre 1 correspondant.
   */
  void setOrder(Int16 order, ItemTypeId linear_type);

  /*!
   * \brief Ajoute une arête à la liste des arêtes
   *
   * \a n0 noeud origine
   * \a n1 noeud extrémité
   * \a f_left numéro local de la face à gauche
   * \a f_right numéro local de la face à droite
   */
  void addEdge(Integer edge_index, Integer n0, Integer n1, Integer f_left, Integer f_right);

  /*!
   * \brief Ajoute une arête et une face
   *
   * \a edge_face_index index local de l'arête et de la face.
   * \a begin_end_node couple (noeud origine,noeud extrémité) de l'arête et la face à ajouter.
   * \a left_and_right_face couple numéro local (face à gauche, face à droite) de l'arête à ajouter
   */
  void addEdgeAndFaceLine(Int32 edge_face_index,
                          std::array<Int16, 2> begin_end_node,
                          std::array<Int16, 2> left_and_right_face);

  /*!
   * \brief Ajoute une arête pour une maille 2D d'un maillage en 3D.
   *
   * \a n0 noeud origine
   * \a n1 noeud extrémité
   */
  void addEdge2D(Integer edge_index, Integer n0, Integer n1);

  //! Ajoute un sommet à la liste des faces (pour les elements 1D)
  void addFaceVertex(Integer face_index, Integer n0);

  //! Ajoute une ligne à la liste des faces (pour les elements 2D)
  void addFaceLine(Integer face_index, Integer n0, Integer n1);

  //! Ajoute une ligne quadratique à la liste des faces (pour les elements 2D)
  void addFaceLine3(Integer face_index, Integer n0, Integer n1, Integer n2);

  //! Ajoute un triangle à la liste des faces
  void addFaceTriangle(Integer face_index, Integer n0, Integer n1, Integer n2);

  //! Ajoute un triangle quadratique à la liste des faces
  void addFaceTriangle6(Integer face_index, Integer n0, Integer n1, Integer n2,
                        Integer n3, Integer n4, Integer n5);

  //! Ajoute un quadrilatère à la liste des faces
  void addFaceQuad(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3);

  //! Ajoute un quadrilatère quadratique à la liste des faces
  void addFaceQuad8(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                    Integer n4, Integer n5, Integer n6, Integer n7);

  //! Ajoute un quadrilatère quadratique à la liste des faces
  void addFaceQuad9(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                    Integer n4, Integer n5, Integer n6, Integer n7, Integer n8);

  //! Ajoute un pentagone à la liste des faces
  void addFacePentagon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3, Integer n4);

  //! Ajoute un hexagone à la liste des faces
  void addFaceHexagon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                      Integer n4, Integer n5);

  //! Ajoute un heptagone à la liste des faces
  void addFaceHeptagon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                       Integer n4, Integer n5, Integer n6);

  //! Ajoute un heptagone à la liste des faces
  void addFaceOctogon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                      Integer n4, Integer n5, Integer n6, Integer n7);

  //! Ajoute une face générique à la liste des faces
  void addFaceGeneric(Integer face_index, Integer type_id, ConstArrayView<Integer> n);

  //! Calcule les relations face->arêtes
  void computeFaceEdgeInfos();

  //! Positionne l'information indiquant si le type est valide pour une maille
  void setIsValidForCell(bool is_valid)
  {
    m_is_valid_for_cell = is_valid;
  }

  //! Positionne l'information indiquant si le type a un nœud au centre
  void setHasCenterNode(bool has_center_node)
  {
    m_has_center_node = has_center_node;
  }

 private:

  void _setNbEdgeAndFace(Integer nb_edge,Integer nb_face);
  void _checkDimension(Int16 dim);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

