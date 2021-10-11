﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializedData.cc                                           (C) 2000-2021 */
/*                                                                           */
/* Donnée sérialisée.                                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISerializedData.h"

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/Array.h"

#include "arcane/ISerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée sérialisée.
 */
class SerializedData
: public ReferenceCounterImpl
, public ISerializedData
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  SerializedData();
  SerializedData(eDataType base_data_type,Int64 memory_size,
                 Integer nb_dimension,Int64 nb_element,Int64 nb_base_element,
                 bool is_multi_size,Int64ConstArrayView extents);

 public:

  eDataType baseDataType() const override { return m_base_data_type; }
  Integer nbDimension() const override { return m_nb_dimension; }
  Int64 nbElement() const override { return m_nb_element; }
  bool isMultiSize() const override { return m_is_multi_size; }
  Int64 memorySize() const override { return m_memory_size; }
  Int64ConstArrayView extents() const override { return m_extents; }
  Int64 nbBaseElement() const override { return m_nb_base_element; }
  ByteConstArrayView buffer() const override { return m_const_buffer.constSmallView(); }
  ByteArrayView buffer() override { return m_buffer.smallView(); }
  Span<const Byte> bytes() const override { return m_const_buffer; }
  Span<const Byte> constBytes() const override { return m_const_buffer; }
  Span<Byte> bytes() override { return m_buffer; }
  void setBuffer(ByteArrayView buffer) override;
  void setBuffer(ByteConstArrayView buffer) override;
  void setBytes(Span<Byte> bytes) override { setWritableBytes(bytes); }
  void setBytes(Span<const Byte> bytes) override { setConstBytes(bytes); }
  Span<Byte> writableBytes() override { return m_buffer; }
  void setWritableBytes(Span<Byte> bytes) override;
  void setConstBytes(Span<const Byte> bytes) override;
  void allocateMemory(Int64 size) override;

 public:

  void serialize(ISerializer* buffer) override;
  void serialize(ISerializer* buffer) const override;

 public:

  void computeHash(IHashAlgorithm* algo,ByteArray& output) const override;

 private:

  eDataType m_base_data_type;
  Int64 m_memory_size;
  Integer m_nb_dimension;
  Int64 m_nb_element;
  Int64 m_nb_base_element;
  bool m_is_multi_size;
  // TODO: supprimer le champs 'm_dimensions' mais cela implique de
  // changer la valeur de computeHash() donc à voir le meilleur moment
  // pour le faire.
  UniqueArray<Int32> m_dimensions;
  UniqueArray<Int64> m_extents;
  Int64 m_element_size;
  Span<Byte> m_buffer;
  Span<const Byte> m_const_buffer;
  UniqueArray<Byte> m_stored_buffer;

 private:

  void _serialize(ISerializer* sbuf) const;
  void _copyExtentsToDimensions();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SerializedData::
SerializedData()
: m_base_data_type(DT_Unknown)
, m_memory_size(0)
, m_nb_dimension(0)
, m_nb_element(0)
, m_nb_base_element(0)
, m_is_multi_size(false)
, m_element_size(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SerializedData::
SerializedData(eDataType base_data_type,Int64 memory_size,
               Integer nb_dimension,Int64 nb_element,Int64 nb_base_element,
               bool is_multi_size,Int64ConstArrayView extents)
: m_base_data_type(base_data_type)
, m_memory_size(memory_size)
, m_nb_dimension(nb_dimension)
, m_nb_element(nb_element)
, m_nb_base_element(nb_base_element)
, m_is_multi_size(is_multi_size)
, m_extents(extents)
, m_element_size(dataTypeSize(m_base_data_type))
{
  _copyExtentsToDimensions();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
_copyExtentsToDimensions()
{
  Integer n = m_extents.size();
  m_dimensions.resize(n);
  // Il ne faut pas lever d'exceptions si on dépasse les bornes sinon
  // le code lèvera une exception dès que le nombre d'éléments du tableau
  // dépasse 32 bits. Cela n'est pas très grave si les valeurs de 'm_dimensions'
  // ne sont pas valide car ce n'est plus utilisé que dans computeHash() pour
  // garder la valeur compatible.
  for( Integer i=0; i<n; ++i )
    m_dimensions[i] = static_cast<Int32>(m_extents[i]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
setBuffer(ByteArrayView buffer)
{
  setBytes(Span<Byte>(buffer));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
setBuffer(ByteConstArrayView buffer)
{
  setBytes(Span<const Byte>(buffer));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
setWritableBytes(Span<Byte> buffer)
{
  m_buffer = buffer;
  m_const_buffer = buffer;
  m_stored_buffer.clear();
  m_memory_size = buffer.size();

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
setConstBytes(Span<const Byte> buffer)
{
  m_const_buffer = buffer;
  m_buffer = Span<Byte>();
  m_stored_buffer.clear();
  m_memory_size = buffer.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
allocateMemory(Int64 size)
{
  m_stored_buffer = UniqueArray<Byte>(size);
  m_buffer = m_stored_buffer;
  m_const_buffer = m_stored_buffer.view();
  m_memory_size = size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
computeHash(IHashAlgorithm* algo,ByteArray& output) const
{
  // TODO: faire avec le support 64 bits mais cela change le hash.
  algo->computeHash(m_const_buffer.constSmallView(),output);
  const Byte* ptr = reinterpret_cast<const Byte*>(m_dimensions.data());
  Integer msize = CheckedConvert::multiply(m_dimensions.size(),(Integer)sizeof(Integer));
  ByteConstArrayView dim_bytes(msize,ptr);
  algo->computeHash(dim_bytes,output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo ne pas utiliser le type DT_Byte pour la serialisation mais
 * le vrai type de base: ce type peut etre utiliser avec MPI et dans ce
 * cas, si les machines sont heterogenes, on perd l'information du type
 * et le put peut ne pas correspondre.
 */
void SerializedData::
serialize(ISerializer* sbuf) const
{
  _serialize(sbuf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SerializedData::
serialize(ISerializer* sbuf)
{
  ISerializer::eMode mode = sbuf->mode();

  switch(mode){
  case ISerializer::ModeReserve:
    _serialize(sbuf);
    break;
  case ISerializer::ModePut:
    _serialize(sbuf);
    break;
  case ISerializer::ModeGet:
    switch(sbuf->readMode()){
    case ISerializer::ReadReplace:
      {
        m_base_data_type = (eDataType)sbuf->getInteger(); // Pour le m_base_data_type
        m_memory_size = sbuf->getInt64(); // Pour le m_memory_size
        m_nb_dimension = sbuf->getInteger(); // Pour le m_nb_dimension
        m_nb_element = sbuf->getInt64(); // Pour le m_nb_element
        m_nb_base_element = sbuf->getInt64(); // Pour le m_nb_base_element
        m_is_multi_size = sbuf->getInteger(); // Pour le m_is_multi_size
        m_element_size = sbuf->getInt64(); // Pour le m_element_size

        Int64 dimensions_size = sbuf->getInt64();
        m_extents.resize(dimensions_size);
        sbuf->getSpan(m_extents);
        _copyExtentsToDimensions();

        Int64 buffer_size = sbuf->getInt64();
        m_stored_buffer.resize(buffer_size);
        sbuf->getSpan(m_stored_buffer); // Pour les données
        m_buffer = m_stored_buffer;
        m_const_buffer = m_buffer;
      }
      break;
    case ISerializer::ReadAdd:
      ARCANE_THROW(NotImplementedException,"ReadAdd");
      break;
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo ne pas utiliser le type DT_Byte pour la serialisation mais
 * le vrai type de base: ce type peut etre utiliser avec MPI et dans ce
 * cas, si les machines sont heterogenes, on perd l'information du type
 * et le put peut ne pas correspondre.
 */
void SerializedData::
_serialize(ISerializer* sbuf) const
{
  ISerializer::eMode mode = sbuf->mode();
  if (m_extents.size()!=m_dimensions.size())
    ARCANE_FATAL("Incoherence between extents ({0}) and dimensions ({1})",
                 m_extents.size(),m_dimensions.size());
  switch(mode){
  case ISerializer::ModeReserve:
    sbuf->reserveInteger(1); // Pour le m_base_data_type
    sbuf->reserve(DT_Int64,1); // Pour le m_memory_size
    sbuf->reserveInteger(1); // Pour le m_nb_dimension
    sbuf->reserve(DT_Int64,1); // Pour le m_nb_element
    sbuf->reserve(DT_Int64,1); // Pour le m_nb_base_element
    sbuf->reserveInteger(1); // Pour le m_is_multi_size
    sbuf->reserve(DT_Int64,1); // Pour le m_element_size

    sbuf->reserve(DT_Int64,1); // Pour le m_extents.size()
    sbuf->reserveSpan(DT_Int64,m_extents.size()); // Pour les dimensions

    sbuf->reserve(DT_Int64,1); // Pour le m_const_buffer.size()
    sbuf->reserveSpan(DT_Byte,m_const_buffer.size()); // Pour les données
    break;
  case ISerializer::ModePut:
    sbuf->putInteger(m_base_data_type); // Pour le m_base_data_type
    sbuf->putInt64(m_memory_size); // Pour le m_memory_size
    sbuf->putInteger(m_nb_dimension); // Pour le m_nb_dimension
    sbuf->putInt64(m_nb_element); // Pour le m_nb_element
    sbuf->putInt64(m_nb_base_element); // Pour le m_nb_base_element
    sbuf->putInteger(m_is_multi_size); // Pour le m_is_multi_size
    sbuf->putInt64(m_element_size); // Pour le m_element_size

    sbuf->putInt64(m_extents.size()); // Pour le m_extents.size()
    sbuf->putSpan(m_extents); // Pour les dimensions

    sbuf->putInt64(m_const_buffer.size()); // Pour le m_const_buffer.size()
    sbuf->putSpan(m_const_buffer); // Pour les données
    break;
  case ISerializer::ModeGet:
    ARCANE_FATAL("This methode does not handle ModeGet");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateSerializedDataRef(eDataType data_type,Int64 memory_size,
                              Integer nb_dim,Int64 nb_element,Int64 nb_base_element,
                              bool is_multi_size,Int64ConstArrayView dimensions)
{
  auto* x = new SerializedData(data_type,memory_size,nb_dim,nb_element,
                               nb_base_element,is_multi_size,dimensions);
  return makeRef(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateEmptySerializedDataRef()
{
  return makeRef(new SerializedData());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
