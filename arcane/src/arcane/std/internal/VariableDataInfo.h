// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableDataInfo.h                                          (C) 2000-2024 */
/*                                                                           */
/* Informations sur les données d'une variable.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_VARIABLEDATAINFO_H
#define ARCANE_STD_INTERNAL_VARIABLEDATAINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/ArrayShape.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/XmlNode.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur les données d'une variable.
 */
class VariableDataInfo
{
 public:

  VariableDataInfo(const String& full_name, const ISerializedData* sdata);
  VariableDataInfo(const String& full_name, const XmlNode& element);

 public:

  const String& fullName() const { return m_full_name; }
  Integer nbDimension() const { return m_nb_dimension; }
  Int64 dim1Size() const { return m_dim1_size; }
  Int64 dim2Size() const { return m_dim2_size; }
  Int64 nbElement() const { return m_nb_element; }
  Int64 nbBaseElement() const { return m_nb_base_element; }
  Integer dimensionArraySize() const { return m_dimension_array_size; }
  bool isMultiSize() const { return m_is_multi_size; }
  eDataType baseDataType() const { return m_base_data_type; }
  Int64 memorySize() const { return m_memory_size; }
  const ArrayShape& shape() const { return m_shape; }
  void setFileOffset(Int64 v) { m_file_offset = v; }
  Int64 fileOffset() const { return m_file_offset; }

 public:

  void write(XmlNode element) const;

 private:

  static void _addAttribute(XmlNode& node, const String& attr_name, Int64 value)
  {
    node.setAttrValue(attr_name, String::fromNumber(value));
  }

  static void _addAttribute(XmlNode& node, const String& attr_name, const String& value)
  {
    node.setAttrValue(attr_name, value);
  }

  static Integer _readInteger(const XmlNode& node, const String& attr_name)
  {
    return node.attr(attr_name, true).valueAsInteger(true);
  }

  static Int64 _readInt64(const XmlNode& node, const String& attr_name)
  {
    return node.attr(attr_name, true).valueAsInt64(true);
  }

  static bool _readBool(const XmlNode& node, const String& attr_name)
  {
    return node.attr(attr_name, true).valueAsBoolean(true);
  }

  static String _readString(const XmlNode& node, const String& attr_name)
  {
    return node.attr(attr_name, true).value();
  }

 private:

  String m_full_name;
  Integer m_nb_dimension = 0;
  Int64 m_dim1_size = 0;
  Int64 m_dim2_size = 0;
  Int64 m_nb_element = 0;
  Int64 m_nb_base_element = 0;
  Integer m_dimension_array_size = 0;
  bool m_is_multi_size = false;
  eDataType m_base_data_type = DT_Unknown;
  Int64 m_memory_size = 0;
  Int64 m_file_offset = 0;
  ArrayShape m_shape;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tableau associatif des données des variables.
 */
class VariableDataInfoMap
{
  using MapType = std::map<String, Ref<VariableDataInfo>>;

 public:

  using const_iterator = MapType::const_iterator;

 public:

  //! Ajoute une variable
  Ref<VariableDataInfo> add(const String& full_name, const ISerializedData* sdata);
  //! Ajoute une variable
  Ref<VariableDataInfo> add(const String& full_name, const XmlNode& node);

  //! Retourne la variable de nom \a full_name. Retourne null si non trouvé.
  Ref<VariableDataInfo> find(const String& full_name) const;

  //@{ //! Itérateurs
  const_iterator begin() const { return m_data_info_map.begin(); }
  const_iterator end() const { return m_data_info_map.end(); }
  //@}

 private:

  MapType m_data_info_map;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
