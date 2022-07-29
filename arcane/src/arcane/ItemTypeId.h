// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeId.h                                                (C) 2000-2022 */
/*                                                                           */
/* Type d'une entité.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMTYPEID_H
#define ARCANE_ITEMTYPEID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Type d'une entité (Item).
 */
class ARCANE_CORE_EXPORT ItemTypeId
{
 public:
  constexpr ARCCORE_HOST_DEVICE ItemTypeId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit ItemTypeId(Int16 id) : m_type_id(id){}
  constexpr ARCCORE_HOST_DEVICE operator Int16() const { return m_type_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInt16() const { return m_type_id; }
 public:
  constexpr ARCCORE_HOST_DEVICE Int16 typeId() const { return m_type_id; }
  constexpr ARCCORE_HOST_DEVICE bool isNull() const { return m_type_id==IT_NullType; }
 private:
  Int16 m_type_id = IT_NullType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
