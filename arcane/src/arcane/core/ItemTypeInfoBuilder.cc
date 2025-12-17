// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeInfoBuilder.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Constructeur de type d'entité de maillage.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypeInfoBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
_checkDimension(Int16 dim)
{
  if (dim < 0)
    return;
  if (m_dimension < 0)
    m_dimension = dim;
  if (m_dimension != dim)
    ARCANE_FATAL("Incoherent dimension for ItemType: current='{0}' new='{1}'",
                 m_dimension, dim);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
setInfos(ItemTypeMng* mng, ItemTypeId type_id, String type_name,
         Dimension dimension, Int32 nb_node, Int32 nb_edge, Int32 nb_face)
{
  m_mng = mng;
  m_type_id = type_id;
  m_linear_type_id = type_id;
  m_dimension = static_cast<Int16>(dimension);
  m_nb_node = nb_node;
  m_type_name = type_name;
  _setNbEdgeAndFace(nb_edge, nb_face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
setInfos(ItemTypeMng* mng, ItemTypeId type_id, String type_name,
         Int32 nb_node, Int32 nb_edge, Int32 nb_face)
{
  setInfos(mng, type_id, type_name, Dimension::DimUnknown, nb_node, nb_edge, nb_face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
setInfos(ItemTypeMng* mng, Int16 type_id, String type_name,
         Int32 nb_node, Int32 nb_edge, Int32 nb_face)
{
  setInfos(mng, ItemTypeId::fromInteger(type_id), type_name, nb_node, nb_edge, nb_face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
setInfos(ItemTypeMng* mng, Int16 type_id, String type_name, Dimension dimension,
         Int32 nb_node, Int32 nb_edge, Int32 nb_face)
{
  setInfos(mng, ItemTypeId(type_id), type_name, dimension, nb_node, nb_edge, nb_face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
setOrder(Int16 order, ItemTypeId linear_type_id)
{
  m_order = order;
  m_linear_type_id = linear_type_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addEdge(Int32 edge_index, Int32 n0, Int32 n1, Int32 f_left, Int32 f_right)
{
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + edge_index] = buf.size();
  buf.add(n0);
  buf.add(n1);
  buf.add(f_left);
  buf.add(f_right);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addEdge2D(Integer edge_index, Integer n0, Integer n1)
{
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + edge_index] = buf.size();
  buf.add(n0);
  buf.add(n1);
  buf.add(-1);
  buf.add(-1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceVertex(Integer face_index, Integer n0)
{
  _checkDimension(1);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_FaceVertex);
  buf.add(1);
  buf.add(n0);
  buf.add(0); // no edge
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceLine(Integer face_index, Integer n0, Integer n1)
{
  _checkDimension(2);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Line2);
  buf.add(2);
  buf.add(n0);
  buf.add(n1);
  buf.add(0); // no edge
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addEdgeAndFaceLine(Int32 edge_face_index,
                   std::array<Int16, 2> begin_end_node,
                   std::array<Int16, 2> left_and_right_face)
{
  Int16 n0 = begin_end_node[0];
  Int16 n1 = begin_end_node[1];
  addFaceLine(edge_face_index, n0, n1);
  addEdge(edge_face_index, n0, n1, left_and_right_face[0], left_and_right_face[1]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Ajoute une ligne quadratique à la liste des faces (pour les elements 2D)
void ItemTypeInfoBuilder::
addFaceLine3(Integer face_index, Integer n0, Integer n1, Integer n2)
{
  _checkDimension(2);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Line3);
  buf.add(3);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(0); // no edge
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Ajoute une ligne quadratique à la liste des faces (pour les elements 2D)
void ItemTypeInfoBuilder::
addFaceLine4(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3)
{
  _checkDimension(2);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Line4);
  buf.add(4);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(n3);
  buf.add(0); // no edge
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceTriangle(Integer face_index, Integer n0, Integer n1, Integer n2)
{
  _checkDimension(3);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Triangle3);
  buf.add(3);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(3);
  for (Integer i = 0; i < 3; ++i)
    buf.add(-1); // undef value, filled by ItemTypeInfoBuilder::computeFaceEdgeInfos
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceTriangle6(Integer face_index, Integer n0, Integer n1, Integer n2,
                 Integer n3, Integer n4, Integer n5)
{
  _checkDimension(3);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Triangle6);
  buf.add(6);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(n3);
  buf.add(n4);
  buf.add(n5);
  buf.add(3);
  for (Integer i = 0; i < 3; ++i)
    buf.add(-1); // undef value, filled by ItemTypeInfoBuilder::computeFaceEdgeInfos
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceQuad(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3)
{
  _checkDimension(3);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Quad4);
  buf.add(4);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(n3);
  buf.add(4);
  for (Integer i = 0; i < 4; ++i)
    buf.add(-1); // undef value, filled by ItemTypeInfoBuilder::computeFaceEdgeInfos
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceQuad8(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
             Integer n4, Integer n5, Integer n6, Integer n7)
{
  _checkDimension(3);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Quad8);
  buf.add(8);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(n3);
  buf.add(n4);
  buf.add(n5);
  buf.add(n6);
  buf.add(n7);
  buf.add(4);
  for (Integer i = 0; i < 4; ++i)
    buf.add(-1); // undef value, filled by ItemTypeInfoBuilder::computeFaceEdgeInfos
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceQuad9(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
             Integer n4, Integer n5, Integer n6, Integer n7, Integer n8)
{
  _checkDimension(3);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Quad9);
  buf.add(9);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(n3);
  buf.add(n4);
  buf.add(n5);
  buf.add(n6);
  buf.add(n7);
  buf.add(n8);
  buf.add(4);
  for (Integer i = 0; i < 4; ++i)
    buf.add(-1); // undef value, filled by ItemTypeInfoBuilder::computeFaceEdgeInfos
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFacePentagon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3, Integer n4)
{
  _checkDimension(3);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Pentagon5);
  buf.add(5);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(n3);
  buf.add(n4);
  buf.add(5);
  for (Integer i = 0; i < 5; ++i)
    buf.add(-1); // undef value, filled by ItemTypeInfoBuilder::computeFaceEdgeInfos
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceHexagon(Integer face_index, Integer n0, Integer n1, Integer n2,
               Integer n3, Integer n4, Integer n5)
{
  _checkDimension(3);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Hexagon6);
  buf.add(6);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(n3);
  buf.add(n4);
  buf.add(n5);
  buf.add(6);
  for (Integer i = 0; i < 6; ++i)
    buf.add(-1); // undef value, filled by ItemTypeInfoBuilder::computeFaceEdgeInfos
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceHeptagon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                Integer n4, Integer n5, Integer n6)
{
  _checkDimension(3);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Heptagon7);
  buf.add(7);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(n3);
  buf.add(n4);
  buf.add(n5);
  buf.add(n6);
  buf.add(7);
  for (Integer i = 0; i < 7; ++i)
    buf.add(-1); // undef value, filled by ItemTypeInfoBuilder::computeFaceEdgeInfos
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceOctogon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
               Integer n4, Integer n5, Integer n6, Integer n7)
{
  _checkDimension(3);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(IT_Octogon8);
  buf.add(8);
  buf.add(n0);
  buf.add(n1);
  buf.add(n2);
  buf.add(n3);
  buf.add(n4);
  buf.add(n5);
  buf.add(n6);
  buf.add(n7);
  buf.add(8);
  for (Integer i = 0; i < 8; ++i)
    buf.add(-1); // undef value, filled by ItemTypeInfoBuilder::computeFaceEdgeInfos
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
addFaceGeneric(Integer face_index, Integer type_id, ConstArrayView<Integer> n)
{
  // Actuellement cette méthode est uniquement appelée pour les faces 2D donc
  // l'élément actuel doit être de dimension 3.
  _checkDimension(3);
  Array<Integer>& buf = m_mng->m_ids_buffer;
  buf[m_first_item_index + m_nb_edge + face_index] = buf.size();
  buf.add(type_id);
  Integer face_nb_node = n.size();
  buf.add(face_nb_node);
  for (Integer i = 0; i < face_nb_node; ++i)
    buf.add(n[i]);
  buf.add(face_nb_node); // nb edge; ne traite pas de cas particulier pour n==2
  for (Integer i = 0; i < face_nb_node; ++i)
    buf.add(-1); // undef value, filled by ItemTypeInfoBuilder::computeFaceEdgeInfos
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
computeFaceEdgeInfos()
{
  Integer cell_nb_edge = nbLocalEdge();
  for (Integer i_face = 0; i_face < nbLocalFace(); ++i_face) {
    Array<Integer>& buf = m_mng->m_ids_buffer;
    Integer fi = buf[m_first_item_index + m_nb_edge + i_face];
    Integer* index = &buf[fi];
    LocalFace local_face(index);
    Integer face_nb_node = local_face.nbNode();
    Integer face_nb_edge = local_face.nbEdge();
    for (Integer i_edge = 0; i_edge < face_nb_edge; ++i_edge) {
      // L'objectif est de trouver l'arête de sommet [i_edge, i_edge+1] dans l'élément
      Integer beginNode = local_face.node(i_edge);
      Integer endNode = local_face.node((i_edge + 1) % face_nb_edge);
      Integer face_edge = -1;
      for (Integer i = 0; i < cell_nb_edge; ++i) {
        LocalEdge local_edge = localEdge(i);
        if ((local_edge.beginNode() == beginNode && local_edge.endNode() == endNode) ||
            (local_edge.beginNode() == endNode && local_edge.endNode() == beginNode)) {
          if (face_edge != -1)
            ARCANE_FATAL("Conflicting item definition : duplicated edge [{0}:{1}] found as edge {2} and {3} of item {4}({5})",
                         beginNode, endNode, face_edge, i, typeName(), typeId());
          face_edge = i;
        }
      }
      if (face_edge == -1)
        ARCANE_FATAL("Undefined edge [{0}:{1}] found as edge of item {2}({3})",
                     beginNode, endNode, typeName(), typeId());
      index[3 + face_nb_node + i_edge] = face_edge;
      ARCANE_ASSERT((face_edge == local_face.edge(i_edge)), ("Inconsitent face-edge allocation"));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeInfoBuilder::
_setNbEdgeAndFace(Integer nb_edge, Integer nb_face)
{
  m_nb_face = nb_face;
  m_nb_edge = nb_edge;
  Integer total = m_nb_face + m_nb_edge;
  if (total != 0) {
    Array<Integer>& buf = m_mng->m_ids_buffer;
    m_first_item_index = buf.size();
    buf.resize(m_first_item_index + total);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

