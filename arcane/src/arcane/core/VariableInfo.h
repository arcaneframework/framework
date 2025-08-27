// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableInfo.h                                              (C) 2000-2025 */
/*                                                                           */
/* Infos caractérisant une variable.                                         */
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
 * \brief Infos caractérisant une variable.
 */
class ARCANE_CORE_EXPORT VariableInfo
{
 public:

  ARCCORE_DEPRECATED_2020("Use overload with 'DataStorageTypeInfo' argument")
  VariableInfo(const String& local_name,const String& item_family_name,
               const String& item_group_name,
               const String& mesh_name,eItemKind item_kind,
               Integer dimension,Integer multi_tag,eDataType type);

  VariableInfo(const String& local_name,const String& item_family_name,
               const String& item_group_name,
               const String& mesh_name,
               const VariableTypeInfo& var_type_info,
               const DataStorageTypeInfo& storage_info);

 public:

  //! Nom de la variable
  const String& localName() const { return m_local_name; }
  //! Nom de la famille d'entité à laquelle la variable est associée
  const String& itemFamilyName() const { return m_item_family_name; }
  //! Nom du groupe d'entité à laquelle la variable est associée
  const String& itemGroupName() const { return m_item_group_name; }
  //! Nom du maillage auquel la variable est associée
  const String& meshName() const { return m_mesh_name; }
  //! Nom complet de la variable (associé à la famille)
  const String& fullName() const
  {
    if (m_full_name.null())
      _computeFullName();
    return m_full_name;
  }
  //! Type d'entité de maillage
  eItemKind itemKind() const { return m_variable_type_info.itemKind(); }
  //! Dimension
  Integer dimension() const { return m_variable_type_info.dimension(); }
  //! Multi-tag
  Integer multiTag() const { return m_variable_type_info.multiTag(); }
  //! Type des éléments
  eDataType dataType() const { return m_variable_type_info.dataType(); }
  //! Indique si la variable est partielle
  bool isPartial() const { return m_variable_type_info.isPartial(); }

  /*!
   * \brief Si null, change itemGroupName() en le nom du groupe
   * de toutes les entités de la famille.
   */
  void setDefaultItemGroupName();

  //! Informations sur le type de la variable.
  VariableTypeInfo variableTypeInfo() const { return m_variable_type_info; }
  //! Informations sur le type de conteneur de la variable
  DataStorageTypeInfo storageTypeInfo() const { return m_storage_type_info; }

 public:

  static DataStorageTypeInfo
  _internalGetStorageTypeInfo(eDataType data_type,Integer dimension,Integer multi_tag);

 private:

  //! Nom de la variable
  String m_local_name;
  //! Nom de la famille d'entité à laquelle la variable est associée
  String m_item_family_name;
  //! Nom du groupe d'entité à laquelle la variable est associée
  String m_item_group_name;
  //! Nom du maillage auquel la variable est associée
  String m_mesh_name;
  //! Nom complet de la variable (associé à la famille)
  mutable String m_full_name;
  //! Informations sur le type de la variable.
  VariableTypeInfo m_variable_type_info;
  //! Informations sur le conteneur de donnée de la variable
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

