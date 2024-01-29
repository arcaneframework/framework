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

#include "arcane/std/internal/VariableDataInfo.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/JSONReader.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/ISerializedData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

// TODO: utiliser version avec exception pour la lecture JSON si les
// conversions ne sont pas valides (par exemple on attend un réel et on a
// une chaîne de caractère.

namespace
{
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

  static void _addAttribute(JSONWriter& o, const String& attr_name, Int64 value)
  {
    o.write(attr_name, value);
  }

  static void _addAttribute(JSONWriter& o, const String& attr_name, Int32 value)
  {
    o.write(attr_name, value);
  }

  static void _addAttribute(JSONWriter& o, const String& attr_name, const String& value)
  {
    o.write(attr_name, value);
  }

  static void _addAttribute(JSONWriter& o, const String& attr_name, bool value)
  {
    o.write(attr_name, value);
  }

  static Int32 _readInteger(const JSONValue& jvalue, const String& attr_name)
  {
    return jvalue.expectedChild(attr_name).valueAsInt32();
  }

  static Int64 _readInt64(const JSONValue& jvalue, const String& attr_name)
  {
    return jvalue.expectedChild(attr_name).valueAsInt64();
  }

  static bool _readBool(const JSONValue& jvalue, const String& attr_name)
  {
    return jvalue.expectedChild(attr_name).valueAsBool();
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableDataInfo::
VariableDataInfo(const String& full_name, const ISerializedData* sdata)
: m_full_name(full_name)
, m_nb_dimension(sdata->nbDimension())
, m_nb_element(sdata->nbElement())
, m_nb_base_element(sdata->nbBaseElement())
, m_is_multi_size(sdata->isMultiSize())
{
  Int64ConstArrayView extents = sdata->extents();

  if (m_nb_dimension == 2 && !m_is_multi_size) {
    m_dim1_size = extents[0];
    m_dim2_size = extents[1];
  }
  m_dimension_array_size = extents.size();
  m_base_data_type = sdata->baseDataType();
  m_memory_size = sdata->memorySize();
  m_shape = sdata->shape();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableDataInfo::
VariableDataInfo(const String& full_name, const XmlNode& element)
: m_full_name(full_name)
{
  m_nb_dimension = _readInteger(element, V_NB_DIMENSION);
  m_dim1_size = _readInt64(element, V_DIM1_SIZE);
  m_dim2_size = _readInt64(element, V_DIM2_SIZE);
  m_nb_element = _readInt64(element, V_NB_ELEMENT);
  m_nb_base_element = _readInt64(element, V_NB_BASE_ELEMENT);
  m_dimension_array_size = _readInteger(element, V_DIMENSION_ARRAY_SIZE);
  m_is_multi_size = _readBool(element, V_IS_MULTI_SIZE);
  m_base_data_type = (eDataType)_readInteger(element, V_BASE_DATA_TYPE);
  m_memory_size = _readInt64(element, V_MEMORY_SIZE);
  m_file_offset = _readInt64(element, V_FILE_OFFSET);
  // L'élément est nul si on repart d'une veille protection (avant Arcane 3.7)
  XmlNode shape_attr = element.attr(V_SHAPE);
  if (!shape_attr.null()) {
    String shape_str = shape_attr.value();
    if (!shape_str.empty()) {
      UniqueArray<Int32> values;
      if (builtInGetValue(values, shape_str))
        ARCANE_FATAL("Can not read values '{0}' for attribute 'shape'", shape_str);
      m_shape.setDimensions(values);
    }
  }
  {
    // L'attribut 'compare-hash' est nul si on repart d'une veille protection (avant Arcane 3.12)
    XmlNode hash_attr = element.attr(V_COMPARISON_HASH);
    if (!hash_attr.null())
      m_comparison_hash_value = hash_attr.value();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableDataInfo::
VariableDataInfo(const String& full_name, const JSONValue& jvalue)
: m_full_name(full_name)
{
  // NOTE: Le format JSON n'est valide qu'à partir de la version 3.12 de Arcane.
  m_nb_dimension = _readInteger(jvalue, V_NB_DIMENSION);
  m_dim1_size = _readInt64(jvalue, V_DIM1_SIZE);
  m_dim2_size = _readInt64(jvalue, V_DIM2_SIZE);
  m_nb_element = _readInt64(jvalue, V_NB_ELEMENT);
  m_nb_base_element = _readInt64(jvalue, V_NB_BASE_ELEMENT);
  m_dimension_array_size = _readInteger(jvalue, V_DIMENSION_ARRAY_SIZE);
  m_is_multi_size = _readBool(jvalue, V_IS_MULTI_SIZE);
  m_base_data_type = (eDataType)_readInteger(jvalue, V_BASE_DATA_TYPE);
  m_memory_size = _readInt64(jvalue, V_MEMORY_SIZE);
  m_file_offset = _readInt64(jvalue, V_FILE_OFFSET);
  // L'élément est nul si on repart d'une veille protection (avant Arcane 3.7)
  {
    String shape_str = jvalue.expectedChild(V_SHAPE).valueAsStringView();
    if (!shape_str.empty()) {
      UniqueArray<Int32> values;
      if (builtInGetValue(values, shape_str))
        ARCANE_FATAL("Can not read values '{0}' for attribute 'shape'", shape_str);
      m_shape.setDimensions(values);
    }
  }
  m_comparison_hash_value = jvalue.expectedChild(V_COMPARISON_HASH).valueAsStringView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableDataInfo::
write(XmlNode element, JSONWriter& writer) const
{
  _write(element);
  _write(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableDataInfo::
_write(XmlNode element) const
{
  _addAttribute(element, V_NB_DIMENSION, m_nb_dimension);
  _addAttribute(element, V_DIM1_SIZE, m_dim1_size);
  _addAttribute(element, V_DIM2_SIZE, m_dim2_size);
  _addAttribute(element, V_NB_ELEMENT, m_nb_element);
  _addAttribute(element, V_NB_BASE_ELEMENT, m_nb_base_element);
  _addAttribute(element, V_DIMENSION_ARRAY_SIZE, m_dimension_array_size);
  _addAttribute(element, V_IS_MULTI_SIZE, (m_is_multi_size) ? 1 : 0);
  _addAttribute(element, V_BASE_DATA_TYPE, (Integer)m_base_data_type);
  _addAttribute(element, V_MEMORY_SIZE, m_memory_size);
  _addAttribute(element, V_FILE_OFFSET, m_file_offset);
  _addAttribute(element, V_SHAPE_SIZE, m_shape.dimensions().size());
  _addAttribute(element, V_COMPARISON_HASH, m_comparison_hash_value);
  {
    String s;
    if (builtInPutValue(m_shape.dimensions().smallView(), s))
      ARCANE_FATAL("Can not write '{0}'", m_shape.dimensions());
    _addAttribute(element, V_SHAPE, s);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableDataInfo::
_write(JSONWriter& writer) const
{
  JSONWriter::Object o(writer, m_full_name);
  _addAttribute(writer, V_NB_DIMENSION, m_nb_dimension);
  _addAttribute(writer, V_DIM1_SIZE, m_dim1_size);
  _addAttribute(writer, V_DIM2_SIZE, m_dim2_size);
  _addAttribute(writer, V_NB_ELEMENT, m_nb_element);
  _addAttribute(writer, V_NB_BASE_ELEMENT, m_nb_base_element);
  _addAttribute(writer, V_DIMENSION_ARRAY_SIZE, m_dimension_array_size);
  _addAttribute(writer, V_IS_MULTI_SIZE, m_is_multi_size);
  _addAttribute(writer, V_BASE_DATA_TYPE, (Integer)m_base_data_type);
  _addAttribute(writer, V_MEMORY_SIZE, m_memory_size);
  _addAttribute(writer, V_FILE_OFFSET, m_file_offset);
  _addAttribute(writer, V_SHAPE_SIZE, m_shape.dimensions().size());
  _addAttribute(writer, V_COMPARISON_HASH, m_comparison_hash_value);
  {
    String s;
    if (builtInPutValue(m_shape.dimensions().smallView(), s))
      ARCANE_FATAL("Can not write '{0}'", m_shape.dimensions());
    _addAttribute(writer, V_SHAPE, s);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<VariableDataInfo> VariableDataInfoMap::
_add(VariableDataInfo* v)
{
  auto vref = makeRef(v);
  m_data_info_map.insert(std::make_pair(v->fullName(), vref));
  return vref;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<VariableDataInfo> VariableDataInfoMap::
add(const String& full_name, const ISerializedData* sdata)
{
  return _add(new VariableDataInfo(full_name, sdata));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<VariableDataInfo> VariableDataInfoMap::
add(const String& full_name, const XmlNode& node)
{
  return _add(new VariableDataInfo(full_name, node));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<VariableDataInfo> VariableDataInfoMap::
add(const String& full_name, const JSONValue& jvalue)
{
  return _add(new VariableDataInfo(full_name, jvalue));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<VariableDataInfo> VariableDataInfoMap::
find(const String& full_name) const
{
  auto ivar = m_data_info_map.find(full_name);
  if (ivar != m_data_info_map.end())
    return ivar->second;
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
