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
  friend class VariableDataInfoMap;

 private:

  static constexpr const char* V_NB_DIMENSION = "nb-dimension";
  static constexpr const char* V_DIM1_SIZE = "dim1-size";
  static constexpr const char* V_DIM2_SIZE = "dim2-size";
  static constexpr const char* V_NB_ELEMENT = "nb-element";
  static constexpr const char* V_NB_BASE_ELEMENT = "nb-base-element";
  static constexpr const char* V_DIMENSION_ARRAY_SIZE = "dimension-array-size";
  static constexpr const char* V_IS_MULTI_SIZE = "is-multi-size";
  static constexpr const char* V_BASE_DATA_TYPE = "base-data-type";
  static constexpr const char* V_MEMORY_SIZE = "memory-size";
  static constexpr const char* V_FILE_OFFSET = "file-offset";
  static constexpr const char* V_SHAPE_SIZE = "shape-size";
  static constexpr const char* V_SHAPE = "shape";
  static constexpr const char* V_COMPARISON_HASH = "comparison-hash";

 private:

  VariableDataInfo(const String& full_name, const ISerializedData* sdata);
  VariableDataInfo(const String& full_name, const XmlNode& element);
  VariableDataInfo(const String& full_name, const JSONValue& jvalue);

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
  void setComparisonHashValue(const String& v) { m_comparison_hash_value = v; }
  const String& comparisonHashValue() const { return m_comparison_hash_value; }

 public:

  void write(XmlNode element,JSONWriter& writer) const;

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
  String m_comparison_hash_value;

 private:

  void _write(XmlNode element) const;
  void _write(JSONWriter& writer) const;
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
  //! Ajoute une variable
  Ref<VariableDataInfo> add(const String& full_name, const JSONValue& jvalue);

  //! Retourne la variable de nom \a full_name. Retourne null si non trouvé.
  Ref<VariableDataInfo> find(const String& full_name) const;

  //@{ //! Itérateurs
  const_iterator begin() const { return m_data_info_map.begin(); }
  const_iterator end() const { return m_data_info_map.end(); }
  //@}

 private:

  MapType m_data_info_map;

 private:

  Ref<VariableDataInfo> _add(VariableDataInfo* v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
