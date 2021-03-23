// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringArrayData.cc                                          (C) 2000-2020 */
/*                                                                           */
/* Donnée de type 'UniqueArray<String>'.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/StringArrayData.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/IHashAlgorithm.h"

#include "arcane/datatype/DataStorageTypeInfo.h"
#include "arcane/datatype/DataStorageBuildInfo.h"

#include "arcane/impl/SerializedData.h"

#include "arcane/ISerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringArrayData::
StringArrayData(const DataStorageBuildInfo& dsbi)
: m_trace(dsbi.traceMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataStorageTypeInfo StringArrayData::
staticStorageTypeInfo()
{
  eBasicDataType bdt = eBasicDataType::Byte;
  Int32 nb_basic_type = 0;
  Int32 dimension = 2;
  Int32 multi_tag = 1;
  return DataStorageTypeInfo(bdt,nb_basic_type,dimension,multi_tag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DataStorageTypeInfo StringArrayData::
storageTypeInfo() const
{
  return staticStorageTypeInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializedData> StringArrayData::
createSerializedDataRef(bool use_basic_type) const
{
  return makeRef(const_cast<ISerializedData*>(createSerializedData(use_basic_type)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ISerializedData* StringArrayData::
createSerializedData(bool use_basic_type) const
{
  ARCANE_UNUSED(use_basic_type);

  // Positionne les dimensions et calcule la taille nécessaire pour sérialiser
  // les valeurs
  Int64 needed_memory = 0;
  Int64 nb_element = m_value.largeSize();
  Int64UniqueArray dimensions(nb_element);
  for (Integer i = 0; i < nb_element; ++i) {
    Span<const Byte> str(m_value[i].bytes());
    Int64 len = str.size();
    needed_memory += len;
    dimensions[i] = len;
  }
  Int64 nb_base_element = needed_memory;
  ISerializedData* sd = new SerializedData(DT_Byte, needed_memory, 2, nb_element,
                                           nb_base_element, true, dimensions);
  sd->allocateMemory(needed_memory);

  // Recopie les valeurs dans le tableau alloué
  Span<Byte> svalues = sd->bytes();
  {
    Int64 index = 0;
    for (Integer i = 0; i < nb_element; ++i) {
      Span<const Byte> str(m_value[i].bytes());
      Int64 len = str.size();
      // TODO: utiliser directement une méthode de copie.
      for (Int64 z = 0; z < len; ++z)
        svalues[index + z] = str[z];
      index += len;
    }
  }
  return sd;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
allocateBufferForSerializedData(ISerializedData* sdata)
{
  if (sdata->baseDataType() != DT_Byte)
    throw ArgumentException(A_FUNCINFO, "Bad serialized type");

  sdata->allocateMemory(sdata->memorySize());
  //m_trace->info() << " ALLOC ARRAY STRING ptr=" << (void*)byte_values.begin();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
assignSerializedData(const ISerializedData* sdata)
{
  if (sdata->baseDataType() != DT_Byte)
    throw ArgumentException(A_FUNCINFO, "Bad serialized type");

  Span<const Byte> byte_values = sdata->bytes();
  //m_trace->info() << " ASSIGN ARRAY STRING ptr=" << (void*)byte_values.begin();
  Int64ConstArrayView dimensions = sdata->extents();
  Integer nb_element = dimensions.size();
  m_value.resize(nb_element);
  Int64 index = 0;
  for (Integer i = 0; i < nb_element; ++i) {
    Int64 len = dimensions[i];
    Span<const Byte> v(&byte_values[index], len);
    m_value[i] = String(v);
    index += len;
    //m_trace->info() << " READ STRING i=" << i << " v=" << m_value[i] << " len=" << len << " index=" << index;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
serialize(ISerializer* sbuf, IDataOperation* operation)
{
  // TODO: tester cette méthode.
  ARCANE_UNUSED(operation);

  ISerializer::eMode mode = sbuf->mode();
  if (mode == ISerializer::ModeReserve) {
    Integer size = m_value.size();
    sbuf->reserveInteger(1);
    for (Integer z = 0; z < size; ++z)
      sbuf->reserve(m_value[z]);
  }
  else if (mode == ISerializer::ModePut) {
    Integer size = m_value.size();
    sbuf->putInteger(size);
    for (Integer z = 0; z < size; ++z)
      sbuf->put(m_value[z]);
  }
  else if (mode == ISerializer::ModeGet) {
    switch (sbuf->readMode()) {
    case ISerializer::ReadReplace: {
      Integer size = sbuf->getInteger();
      m_value.resize(size);
      for (Integer z = 0; z < size; ++z)
        sbuf->get(m_value[z]);
    } break;
    case ISerializer::ReadAdd:
      ARCANE_THROW(NotSupportedException, "option 'ReadAdd'");
    }
  }
  else
    ARCANE_THROW(NotSupportedException, "Invalid mode");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation)
{
  ARCANE_UNUSED(sbuf);
  ARCANE_UNUSED(ids);
  ARCANE_UNUSED(operation);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
setName(const String& name)
{
  ARCANE_UNUSED(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
computeHash(IHashAlgorithm* algo, ByteArray& output) const
{
  // Pour l'instant, il faut passer par une sérialisation.
  // TODO supprimer la sérialisation inutile.
  Ref<ISerializedData> s = createSerializedDataRef(true);
  s->computeHash(algo, output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
copy(const IData* data)
{
  const auto* true_data = dynamic_cast<const DataInterfaceType*>(data);
  if (!true_data)
    ARCANE_THROW(ArgumentException, "Can not cast 'IData' to 'StringArrayData'");
  m_value.copy(true_data->value());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
swapValues(IData* data)
{
  auto* true_data = dynamic_cast<ThatClass*>(data);
  if (!true_data)
    ARCANE_THROW(ArgumentException, "Can not cast 'IData' to 'StringArrayData'");
  m_value.swap(true_data->m_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
visitScalar(IScalarDataVisitor*)
{
  ARCANE_THROW(NotSupportedException, "Can not visit scalar data with array data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
visitArray(IArrayDataVisitor* visitor)
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
visitArray2(IArray2DataVisitor*)
{
  ARCANE_THROW(NotSupportedException, "Can not visit array2 data with array data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringArrayData::
visitMultiArray2(IMultiArray2DataVisitor*)
{
  ARCANE_THROW(NotSupportedException, "Can not visit multiarray2 data with array data");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
