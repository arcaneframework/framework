// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PropertyMng.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire des protections.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/IPropertyMng.h"
#include "arcane/core/Properties.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/Observable.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire des protections.
 */
class PropertyMng
: public TraceAccessor
, public IPropertyMng
{
 public:
  static const Int32 SERIALIZE_VERSION = 1;

 public:
  explicit PropertyMng(ITraceMng* tm);
  ~PropertyMng() override;

 public:
  void build();

 public:
  ITraceMng* traceMng() const override { return TraceAccessor::traceMng(); }

 public:
  PropertiesImpl* getPropertiesImpl(const String& full_name) override;
  void destroyProperties(const Properties& p) override;
  void registerProperties(const Properties& p) override;
  void serialize(ISerializer* serializer) override;
  void writeTo(ByteArray& bytes) override;
  void readFrom(Span<const Byte> bytes) override;
  void print(std::ostream& o) const override;
  IObservable* writeObservable() override { return &m_write_observable; }
  IObservable* readObservable() override { return &m_read_observable; }

 private:
 private:
  typedef std::map<String, Properties> PropertiesMapType;

  PropertiesMapType m_properties_map;
  VariableArrayByte* m_property_values_var;
  AutoDetachObservable m_write_observable;
  AutoDetachObservable m_read_observable;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IPropertyMng*
arcaneCreatePropertyMng(ITraceMng* tm)
{
  auto pm = new PropertyMng(tm);
  pm->build();
  return pm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IPropertyMng>
arcaneCreatePropertyMngReference(ITraceMng* tm)
{
  auto pm = arcaneCreatePropertyMng(tm);
  return makeRef(pm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertyMng::
PropertyMng(ITraceMng* tm)
: TraceAccessor(tm)
, m_property_values_var(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertyMng::
~PropertyMng()
{
  delete m_property_values_var;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyMng::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertiesImpl* PropertyMng::
getPropertiesImpl(const String& full_name)
{
  //info() << "GETTING PROPERTIES name=" << full_name;
  auto v = m_properties_map.find(full_name);
  if (v != m_properties_map.end())
    return v->second.impl();
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyMng::
registerProperties(const Properties& p)
{
  //TODO: vérifier pas encore présent.
  m_properties_map.insert(std::make_pair(p.fullName(), p));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyMng::
destroyProperties(const Properties& p)
{
  auto v = m_properties_map.find(p.fullName());
  if (v != m_properties_map.end())
    m_properties_map.erase(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyMng::
serialize(ISerializer* serializer)
{
  switch (serializer->mode()) {
  case ISerializer::ModeReserve:

  {
    serializer->reserveInt32(1); // SERIALIZE_VERSION
    serializer->reserveInt64(1); // Nombre d'éléments dans la map

    for (auto& v : m_properties_map) {
      serializer->reserve(v.first);
      //info() << "SERIALIZE RESERVE name=" << v->first;
      v.second.serialize(serializer);
    }
  } break;
  case ISerializer::ModePut: {
    serializer->putInt32(SERIALIZE_VERSION); // SERIALIZE_VERSION
    serializer->putInt64(m_properties_map.size()); // Nombre d'éléments dans la map

    for (auto& v : m_properties_map) {
      serializer->put(v.first);
      //info() << "SERIALIZE PUT name=" << v->first;
      v.second.serialize(serializer);
    }
  } break;
  case ISerializer::ModeGet:

    Int64 version = serializer->getInt32();
    if (version != SERIALIZE_VERSION) {
      // La relecture se fait avec une protection issue d'une ancienne version de Arcane
      // et qui n'est pas compatible. Affiche un avertissement et ne fait rien.
      // (NOTE: cela risque quand même de poser problême pour ceux qui utilisent les
      // propriétés donc il faudrait peut-être faire un fatal ?)
      pwarning() << "Can not reading properties from imcompatible checkpoint";
      return;
    }

    Int64 n = serializer->getInt64();
    String name;
    for (Integer i = 0; i < n; ++i) {
      serializer->get(name);
      Properties p(this, name);
      //info() << "SERIALIZE GET name=" << name;
      p.serialize(serializer);
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyMng::
writeTo(ByteArray& bytes)
{
  m_write_observable.notifyAllObservers();

  SerializeBuffer sb;
  sb.setSerializeTypeInfo(true);
  sb.setMode(ISerializer::ModeReserve);
  this->serialize(&sb);
  sb.allocateBuffer();
  sb.setMode(ISerializer::ModePut);
  this->serialize(&sb);

  Span<const Byte> buf_bytes = sb.globalBuffer();
  info(4) << "SaveProperties nb_byte=" << buf_bytes.size();
  bytes.copy(buf_bytes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertyMng::
readFrom(Span<const Byte> bytes)
{
  info(4) << "ReadProperties nb_read_byte=" << bytes.size();

  SerializeBuffer sb;
  sb.setSerializeTypeInfo(true);
  sb.initFromBuffer(bytes);
  this->serialize(&sb);

  m_read_observable.notifyAllObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Affiche les propriétés et leurs valeurs sur le flot \a o.
 */
void PropertyMng::
print(std::ostream& o) const
{
  for (const auto& v : m_properties_map) {
    const Properties& p = v.second;
    o << '[' << p.fullName() << "]\n";
    p.print(o);
    o << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
