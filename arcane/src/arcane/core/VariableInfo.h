// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableInfo.h                                              (C) 2000-2025 */
/*                                                                           */
/* Information characterizing a variable.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEINFO_H
#define ARCANE_CORE_VARIABLEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/datatype/DataStorageTypeInfo.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/VariableTypeInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information characterizing a variable.
 */
class ARCANE_CORE_EXPORT VariableInfo
{
 public:

  ARCCORE_DEPRECATED_2020("Use overload with 'DataStorageTypeInfo' argument")
  VariableInfo(const String& local_name, const String& item_family_name,
               const String& item_group_name,
               const String& mesh_name, eItemKind item_kind,
               Integer dimension, Integer multi_tag, eDataType type);

  VariableInfo(const String& local_name, const String& item_family_name,
               const String& item_group_name,
               const String& mesh_name,
               const VariableTypeInfo& var_type_info,
               const DataStorageTypeInfo& storage_info);

 public:

  //! Name of the variable
  const String& localName() const { return m_local_name; }
  //! Name of the entity family to which the variable is associated
  const String& itemFamilyName() const { return m_item_family_name; }
  //! Name of the entity group to which the variable is associated
  const String& itemGroupName() const { return m_item_group_name; }
  //! Name of the mesh to which the variable is associated
  const String& meshName() const { return m_mesh_name; }
  //! Full name of the variable (associated with the family)
  const String& fullName() const
  {
    if (m_full_name.null())
      _computeFullName();
    return m_full_name;
  }
  //! Mesh entity type
  eItemKind itemKind() const { return m_variable_type_info.itemKind(); }
  //! Dimension
  Integer dimension() const { return m_variable_type_info.dimension(); }
  //! Multi-tag
  Integer multiTag() const { return m_variable_type_info.multiTag(); }
  //! Element type
  eDataType dataType() const { return m_variable_type_info.dataType(); }
  //! Indicates if the variable is partial
  bool isPartial() const { return m_variable_type_info.isPartial(); }

  /*!
   * \brief If null, changes itemGroupName() to the name of the group of all
   * entities in the family.
   */
  void setDefaultItemGroupName();

  //! Information about the variable type.
  VariableTypeInfo variableTypeInfo() const { return m_variable_type_info; }
  //! Information about the variable container type
  DataStorageTypeInfo storageTypeInfo() const { return m_storage_type_info; }

 public:

  static DataStorageTypeInfo
  _internalGetStorageTypeInfo(eDataType data_type, Integer dimension, Integer multi_tag);

 private:

  //! Name of the variable
  String m_local_name;
  //! Name of the entity family to which the variable is associated
  String m_item_family_name;
  //! Name of the entity group to which the variable is associated
  String m_item_group_name;
  //! Name of the mesh to which the variable is associated
  String m_mesh_name;
  //! Full name of the variable (associated with the family)
  mutable String m_full_name;
  //! Information about the variable type.
  VariableTypeInfo m_variable_type_info;
  //! Information about the variable data container
  DataStorageTypeInfo m_storage_type_info;

 private:

  String _defaultFamilyName();
  void _computeFullName() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
