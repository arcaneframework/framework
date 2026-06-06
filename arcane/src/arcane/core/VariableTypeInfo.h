// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableTypeInfo.h                                          (C) 2000-2025 */
/*                                                                           */
/* Information characterizing the type of a variable.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLETYPEINFO_H
#define ARCANE_CORE_VARIABLETYPEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information characterizing the type of a variable.
 *
 * Instances of this class can be used in
 * static constructors. To avoid any issues, this class must not
 * use dynamic allocation.
 */
class ARCANE_CORE_EXPORT VariableTypeInfo
{
 public:

  constexpr VariableTypeInfo(eItemKind item_kind,eDataType data_type,Integer dimension,
                             Integer multi_tag,bool is_partial)
 : m_item_kind(item_kind), m_data_type(data_type), m_dimension(dimension),
   m_multi_tag(multi_tag), m_is_partial(is_partial){}

 public:

  //! Mesh entity type
  constexpr eItemKind itemKind() const { return m_item_kind; }
  //! Dimension
  constexpr Integer dimension() const { return m_dimension; }
  //! Multi-tag
  constexpr Integer multiTag() const { return m_multi_tag; }
  //! Data type of the variable
  constexpr eDataType dataType() const { return m_data_type; }
  //! Indicates if the variable is partial
  constexpr bool isPartial() const { return m_is_partial; }

  //! Full name of the variable type
  String fullName() const;

 public:

  //! Default data container associated with this variable type
  DataStorageTypeInfo _internalDefaultDataStorage() const;

 private:

  //! Kind of mesh entities (can be null)
  eItemKind m_item_kind;

  //! Data type of the variable
  eDataType m_data_type;

  //! Dimension of the variable
  Integer m_dimension;

  //! Tag indicating whether variable-sized arrays are used.
  Integer m_multi_tag;

  //! Indicates if the variable is partial.
  bool m_is_partial;

 private:

  String _buildFullTypeName() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
