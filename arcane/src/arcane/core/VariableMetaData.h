// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableMetaData.h                                           C) 2000-2023 */
/*                                                                           */
/* Metadata on a variable.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLEMETADATA_H
#define ARCANE_CORE_VARIABLEMETADATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Metadata on a variable.
 *
 * The information in this class allows for the reconstruction of a
 * variable.
 *
 * If hash2() is not null, it is used. Otherwise, hash() is used.
 */
class ARCANE_CORE_EXPORT VariableMetaData
{
 public:
  /*!
   * \brief Constructor.
   *
   * Constructs the instance for a variable with name \a base_name,
   * from family \a item_family_name and group \a item_group_name.
   * If the variable is not on a mesh, then \a mesh_name,
   * \a item_family_name and \a item_group_name are null.
   */
  VariableMetaData(const String& base_name,const String& mesh_name,
                   const String& item_family_name,const String& item_group_name,
                   bool is_partial);

 public:

  //! Full name of the variable
  String fullName() const { return m_full_name; }
  //! Base name of the variable
  String baseName() const { return m_base_name; }
  String meshName() const { return m_mesh_name; }
  String itemFamilyName() const { return m_item_family_name; }
  String itemGroupName() const { return m_item_group_name; }
  bool isPartial() const { return m_is_partial; }

  String fullType() const { return m_full_type; }
  void setFullType(const String& v) { m_full_type = v; }

  //! Hash of the variable in hexadecimal format
  String hash() const { return m_hash_str; }
  void setHash(const String& v) { m_hash_str = v; }

  //! Hash of the variable in hexadecimal format
  String hash2() const { return m_hash2_str; }
  void setHash2(const String& v) { m_hash2_str = v; }

  //! Hash version (associated with hash2())
  Int32 hashVersion() const { return m_hash_version; }
  void setHashVersion(Int32 v) { m_hash_version = v; }

  Integer property() const { return m_property; }
  void setProperty(Integer v) { m_property = v; }

  String multiTag() const { return m_multi_tag; }
  void setMultiTag(const String& v) { m_multi_tag = v; }


 private:

  String m_base_name;
  String m_mesh_name;
  String m_item_family_name;
  String m_item_group_name;
  String m_full_type;
  String m_hash_str;
  Integer m_property;
  String m_multi_tag;
  String m_full_name;
  bool m_is_partial;
  String m_hash2_str;
  Int32 m_hash_version = 0;

 private:

  void _buildFullName();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
