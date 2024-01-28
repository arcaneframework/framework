// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicReaderWriter.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Lecture/Ecriture simple.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/internal/BasicReaderWriter.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/ItemGroup.h"
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

String BasicReaderWriterCommon::
_getArcaneDBTag()
{
  return "ArcaneCheckpointRestartDataBase";
}

String BasicReaderWriterCommon::
_getOwnMetatadaFile(const String& path, Int32 rank)
{
  StringBuilder filename = path;
  filename += "/own_metadata_";
  filename += rank;
  filename += ".txt";
  return filename;
}

String BasicReaderWriterCommon::
_getArcaneDBFile(const String& path, Int32 rank)
{
  StringBuilder filename = path;
  filename += "/arcane_db_n";
  filename += rank;
  filename += ".acr";
  return filename;
}

String BasicReaderWriterCommon::
_getBasicVariableFile(Int32 version, const String& path, Int32 rank)
{
  if (version >= 3) {
    return _getArcaneDBFile(path, rank);
  }
  StringBuilder filename = path;
  filename += "/var___MAIN___";
  filename += rank;
  filename += ".txt";
  return filename;
}

String BasicReaderWriterCommon::
_getBasicGroupFile(const String& path, const String& name, Int32 rank)
{
  StringBuilder filename = path;
  filename += "/group_";
  filename += name;
  filename += "_";
  filename += rank;
  filename += ".txt";
  return filename;
}

Ref<IDataCompressor> BasicReaderWriterCommon::
_createDeflater(IApplication* app, const String& name)
{
  ServiceBuilder<IDataCompressor> sf(app);
  Ref<IDataCompressor> bc = sf.createReference(app, name);
  return bc;
}

Ref<IHashAlgorithm> BasicReaderWriterCommon::
_createHashAlgorithm(IApplication* app, const String& name)
{
  ServiceBuilder<IHashAlgorithm> sf(app);
  Ref<IHashAlgorithm> bc = sf.createReference(app, name);
  return bc;
}

void BasicReaderWriterCommon::
_fillUniqueIds(const ItemGroup& group, Array<Int64>& uids)
{
  Integer nb_item = group.size();
  uids.clear();
  uids.reserve(nb_item);
  ENUMERATE_ITEM (iitem, group) {
    Int64 uid = iitem->uniqueId();
    uids.add(uid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String BasicReaderWriterCommon::
_getMetaDataFileName(Int32 rank) const
{
  StringBuilder filename = m_path;
  filename += "/metadata";
  filename += "-";
  filename += rank;
  filename += ".txt";
  return filename;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicReaderWriterCommon::
BasicReaderWriterCommon(IApplication* app, IParallelMng* pm,
                        const String& path, eOpenMode open_mode)
: TraceAccessor(pm->traceMng())
, m_application(app)
, m_parallel_mng(pm)
, m_open_mode(open_mode)
, m_path(path)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
