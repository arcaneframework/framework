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
  * \brief Find or add a type for a generic Polyhedron.
  */
  ItemTypeInfo* findOrAddPolyhedron(Int16 nb_node, ConstArrayView<Int16> faces_nb_nodes, ConstArrayView<Int16> faces_nodes)
  {
    return m_item_type_mng->_findOrAddPolyhedron(nb_node, faces_nb_nodes, faces_nodes);
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
