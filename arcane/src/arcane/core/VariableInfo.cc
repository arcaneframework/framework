// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableInfo.cc                                             (C) 2000-2024 */
/*                                                                           */
/* Information characterizing a variable.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableInfo.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/core/datatype/DataTypeTraits.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

DataStorageTypeInfo VariableInfo::
_internalGetStorageTypeInfo(eDataType data_type, Integer dimension, Integer multi_tag)
{
  Integer nb_basic = -1;
  eBasicDataType basic_data_type = eBasicDataType::Unknown;
  switch (data_type) {
  case DT_Byte:
    nb_basic = DataTypeTraitsT<Byte>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Byte>::basicDataType();
    break;
  case DT_Int8:
    nb_basic = DataTypeTraitsT<Int8>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Int8>::basicDataType();
    break;
  case DT_Int16:
    nb_basic = DataTypeTraitsT<Int16>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Int16>::basicDataType();
    break;
  case DT_Int32:
    nb_basic = DataTypeTraitsT<Int32>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Int32>::basicDataType();
    break;
  case DT_Int64:
    nb_basic = DataTypeTraitsT<Int64>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Int64>::basicDataType();
    break;
  case DT_Real:
    nb_basic = DataTypeTraitsT<Real>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Real>::basicDataType();
    break;
  case DT_Float32:
    nb_basic = DataTypeTraitsT<Float32>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Float32>::basicDataType();
    break;
  case DT_Float16:
    nb_basic = DataTypeTraitsT<Float16>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Float16>::basicDataType();
    break;
  case DT_BFloat16:
    nb_basic = DataTypeTraitsT<BFloat16>::nbBasicType();
    basic_data_type = DataTypeTraitsT<BFloat16>::basicDataType();
    break;
  case DT_Real2:
    nb_basic = DataTypeTraitsT<Real2>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Real2>::basicDataType();
    break;
  case DT_Real3:
    nb_basic = DataTypeTraitsT<Real3>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Real3>::basicDataType();
    break;
  case DT_Real2x2:
    nb_basic = DataTypeTraitsT<Real2x2>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Real2x2>::basicDataType();
    break;
  case DT_Real3x3:
    nb_basic = DataTypeTraitsT<Real3x3>::nbBasicType();
    basic_data_type = DataTypeTraitsT<Real3x3>::basicDataType();
    break;
  case DT_String:
    // For strings, the container contains
    // 'Byte' and the dimension is 1 greater than that
    // of the variable.
    nb_basic = 0;
    multi_tag = 1;
    dimension += 1;
    basic_data_type = eBasicDataType::Byte;
    break;
  case DT_Unknown:
    break;
  }
  return DataStorageTypeInfo(basic_data_type, nb_basic, dimension, multi_tag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableInfo::
VariableInfo(const String& local_name, const String& item_family_name,
             const String& item_group_name,
             const String& mesh_name, eItemKind item_kind,
             Integer dimension, Integer multi_tag, eDataType data_type)
: VariableInfo(local_name, item_family_name, item_group_name, mesh_name,
               VariableTypeInfo(item_kind, data_type, dimension, multi_tag, !item_group_name.null()),
               _internalGetStorageTypeInfo(data_type, dimension, multi_tag))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableInfo::
VariableInfo(const String& local_name, const String& item_family_name,
             const String& item_group_name,
             const String& mesh_name,
             const VariableTypeInfo& var_type_info,
             const DataStorageTypeInfo& storage_info)
: m_local_name(local_name)
, m_item_family_name(item_family_name)
, m_item_group_name(item_group_name)
, m_mesh_name(mesh_name)
, m_variable_type_info(var_type_info)
, m_storage_type_info(storage_info)
{
  if (m_item_family_name.null())
    m_item_family_name = _defaultFamilyName();
  // m_item_group_name can be null here. In this case,
  // it will be initialized later via
  // setDefaultItemGroupName(). It should not be done
  // here because updating the name entails memory allocations/deallocations
  // and this constructor might be called often when searching
  // for variables.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String VariableInfo::
_defaultFamilyName()
{
  String family_name;
  switch (itemKind()) {
  case IK_Particle:
    ARCANE_FATAL("No default family for 'particle' variable '{0}'", m_local_name);
  case IK_DoF:
    ARCANE_FATAL("No default family for 'dof' variable '{0}'", m_local_name);
  case IK_Node:
    family_name = ItemTraitsT<Node>::defaultFamilyName();
    break;
  case IK_Edge:
    family_name = ItemTraitsT<Edge>::defaultFamilyName();
    break;
  case IK_Face:
    family_name = ItemTraitsT<Face>::defaultFamilyName();
    break;
  case IK_Cell:
    family_name = ItemTraitsT<Cell>::defaultFamilyName();
    break;
  case IK_Unknown:
    break;
  }
  return family_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculates the full name of the variable.
 *
 * This name requires concatenating strings and therefore
 * making allocations which can be costly.
 * The name is only calculated if fullName() is explicitly requested.
 */
void VariableInfo::
_computeFullName() const
{
  StringBuilder full_name;
  if (m_mesh_name.null()) {
    full_name = String();
  }
  else {
    full_name = m_mesh_name;
    full_name += "_";
  }

  if (m_item_family_name.null()) {
    full_name += m_local_name;
  }
  else {
    full_name += m_item_family_name;
    full_name += "_";
    full_name += m_local_name;
  }
  m_full_name = full_name.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableInfo::
setDefaultItemGroupName()
{
  if (!m_item_group_name.null())
    return;
  // The name constructed here must be consistent with that in DynamicMeshKindInfos.cc
  m_item_group_name = "All" + m_item_family_name + "s";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
