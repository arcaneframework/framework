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
#include "arcane/utils/Array.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/ISerializedData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

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
  m_nb_dimension = _readInteger(element, "nb-dimension");
  m_dim1_size = _readInt64(element, "dim1-size");
  m_dim2_size = _readInt64(element, "dim2-size");
  m_nb_element = _readInt64(element, "nb-element");
  m_nb_base_element = _readInt64(element, "nb-base-element");
  m_dimension_array_size = _readInteger(element, "dimension-array-size");
  m_is_multi_size = _readBool(element, "is-multi-size");
  m_base_data_type = (eDataType)_readInteger(element, "base-data-type");
  m_memory_size = _readInt64(element, "memory-size");
  m_file_offset = _readInt64(element, "file-offset");
  // L'élément est nul si on repart d'une veille protection (avant Arcane 3.7)
  XmlNode shape_attr = element.attr("shape");
  if (!shape_attr.null()) {
    String shape_str = shape_attr.value();
    if (!shape_str.empty()) {
      UniqueArray<Int32> values;
      if (builtInGetValue(values, shape_str))
        ARCANE_FATAL("Can not read values '{0}' for attribute 'shape'", shape_str);
      m_shape.setDimensions(values);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableDataInfo::
write(XmlNode element) const
{
  _addAttribute(element, "nb-dimension", m_nb_dimension);
  _addAttribute(element, "dim1-size", m_dim1_size);
  _addAttribute(element, "dim2-size", m_dim2_size);
  _addAttribute(element, "nb-element", m_nb_element);
  _addAttribute(element, "nb-base-element", m_nb_base_element);
  _addAttribute(element, "dimension-array-size", m_dimension_array_size);
  _addAttribute(element, "is-multi-size", (m_is_multi_size) ? 1 : 0);
  _addAttribute(element, "base-data-type", (Integer)m_base_data_type);
  _addAttribute(element, "memory-size", m_memory_size);
  _addAttribute(element, "file-offset", m_file_offset);
  _addAttribute(element, "shape-size", m_shape.dimensions().size());
  {
    String s;
    if (builtInPutValue(m_shape.dimensions().smallView(), s))
      ARCANE_FATAL("Can not write '{0}'", m_shape.dimensions());
    _addAttribute(element, "shape", s);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
