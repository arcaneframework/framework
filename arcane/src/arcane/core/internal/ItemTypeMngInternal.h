// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeMngInternal.h                                       (C) 2000-2026 */
/*                                                                           */
/* Internal API for ItemTypeMng.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMTYPEMNGINTERNAL_H
#define ARCANE_CORE_ITEMTYPEMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Array.h"

#include "arcane/core/ItemTypeMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT ItemTypeMngInternal
{
 public:

  explicit ItemTypeMngInternal(ItemTypeMng* iti)
  : m_item_type_mng(iti)
  {}

 public:

  /*!
   * \brief Create a temporary type for generic Polyhedron.
   *
   * \a ids_buffer will be used to store connectivities informations.
   * It must remain valid until the type is no longer used.
   *
   * \param nb_node number of nodes of the polyhedron.
   * \pram faces_nb_node number of nodes of each face.
   * \param faces_nodes list of nodes of the faces. The elements
   * from `0` to `faces_nb_node[0]-1` are the node index of the first face,
   * the elements from `faces_nb_nodes[0]` to `face_nb_nodes[0]+faces_nb_node[1]-1`
   * are the nodes index of the second face and so on.
   */
  ItemTypeInfo* getTemporaryPolyhedron(Array<Int32>* ids_buffer, Int16 nb_node,
                                       ConstArrayView<Int16> faces_nb_node,
                                       ConstArrayView<Int16> faces_nodes)
  {
    return m_item_type_mng->_getTemporaryPolyhedron(ids_buffer, nb_node, faces_nb_node, faces_nodes);
  }

 private:

  ItemTypeMng* m_item_type_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
