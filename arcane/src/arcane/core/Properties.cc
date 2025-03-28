// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Properties.cc                                               (C) 2000-2024 */
/*                                                                           */
/* Liste de propriétés.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Properties.h"

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/datatype/SmallVariant.h"
#include "arcane/core/datatype/DataTypeTraits.h"

#include "arcane/core/IPropertyMng.h"
#include "arcane/core/ISerializer.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PropertyVariant
{
 public:
  
  enum eType
    {
      PV_None = 0,

      PV_ScalarReal = 1,
      PV_ScalarInt32 = 2,
      PV_ScalarInt64 = 3,
      PV_ScalarBool = 4,
      PV_ScalarString = 5,

      PV_ArrayReal = 6,
      PV_ArrayInt32 = 7,
      PV_ArrayInt64 = 8,
      PV_ArrayBool = 9,
      PV_ArrayString = 10,
    };
  static const int NB_TYPE = 11;
 private:

  PropertyVariant()
  : m_int32(0), m_int64(0), m_real(0), m_bool(0), m_string(0), m_is_scalar(false), m_type(PV_None)
  {
  }

 public:
  ~PropertyVariant()
  {
    delete m_int32;
    delete m_int64;
    delete m_real;
    delete m_bool;
    delete m_string;
  }
  static PropertyVariant* create(Int32ConstArrayView v)
  {
    PropertyVariant* p = new PropertyVariant();
    p->m_int32 = new UniqueArray<Int32>(v);
    p->m_type = PV_ArrayInt32;
    return p;
  }

  static PropertyVariant* create(Int64ConstArrayView v)
  {
    PropertyVariant* p = new PropertyVariant();
    p->m_int64 = new UniqueArray<Int64>(v);
    p->m_type = PV_ArrayInt64;
    return p;
  }

  static PropertyVariant* create(RealConstArrayView v)
  {
    PropertyVariant* p = new PropertyVariant();
    p->m_real = new UniqueArray<Real>(v);
    p->m_type = PV_ArrayReal;
    return p;
  }

  static PropertyVariant* create(BoolConstArrayView v)
  {
    PropertyVariant* p = new PropertyVariant();
    p->m_bool = new UniqueArray<bool>(v);
    p->m_type = PV_ArrayBool;
    return p;
  }

  static PropertyVariant* create(StringConstArrayView v)
  {
    PropertyVariant* p = new PropertyVariant();
    p->m_string = new UniqueArray<String>(v);
    p->m_type = PV_ArrayString;
    return p;
  }
    
  static PropertyVariant* create(const SmallVariant& sv)
  {
    PropertyVariant* p = new PropertyVariant();
    p->m_is_scalar = true;
    p->m_scalar = sv;
    switch(sv.type()){
    case SmallVariant::TUnknown: p->m_type = PV_None; break;
    case SmallVariant::TReal: p->m_type = PV_ScalarReal; break;
    case SmallVariant::TInt32: p->m_type = PV_ScalarInt32; break;
    case SmallVariant::TInt64: p->m_type = PV_ScalarInt64; break;
    case SmallVariant::TBool: p->m_type = PV_ScalarBool; break;
    case SmallVariant::TString: p->m_type = PV_ScalarString; break;
    }
    return p;
  }

  SmallVariant* getScalar()
  {
    if (!m_is_scalar)
      return 0;
    return &m_scalar;
  }
  
  UniqueArray<Int32>* get(Int32) const
  {
    return m_int32;
  }

  UniqueArray<Int64>* get(Int64) const
  {
    return m_int64;
  }

  UniqueArray<Real>* get(Real) const
  {
    return m_real;
  }

  UniqueArray<bool>* get(bool) const
  {
    return m_bool;
  }

  UniqueArray<String>* get(const String&) const
  {
    return m_string;
  }
  
  eType type() const { return m_type; }

 private:

  UniqueArray<Int32>* m_int32;
  UniqueArray<Int64>* m_int64;
  UniqueArray<Real>* m_real;
  UniqueArray<bool>* m_bool;
  UniqueArray<String>* m_string;
  bool m_is_scalar;
  eType m_type;
  SmallVariant m_scalar;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IPropertyType
{
 public:
  virtual ~IPropertyType(){}
 public:
  virtual void print(std::ostream& o,PropertyVariant* v) =0;
  virtual const String& typeName() const =0;
  virtual void serializeReserve(ISerializer* s,PropertyVariant* v) =0;
  virtual void serializePut(ISerializer* s,PropertyVariant* v) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

template<typename DataType>
void _directPutScalar(ISerializer* s,const DataType& value)
{
  s->put(value);
}

void _directPutScalar(ISerializer* s,const bool& value)
{
  s->putByte(value);
}

template<typename DataType>
void _directReserveScalar(ISerializer* s,const DataType&)
{
  s->reserve(DataTypeTraitsT<DataType>::basicDataType(),1);
}

void _directReserveScalar(ISerializer* s,const String& value)
{
  s->reserve(value);
}

template<typename DataType>
void _directReserve(ISerializer* s,Span<const DataType> values)
{
  s->reserveInt64(1);
  s->reserveSpan(DataTypeTraitsT<DataType>::basicDataType(),values.size());
}


void _directReserve(ISerializer* s,Span<const bool> values)
{
  Int64 n = values.size();
  s->reserveInt64(1);
  s->reserveSpan(eBasicDataType::Byte,n);
}

void _directReserve(ISerializer* s,Span<const String> values)
{
  s->reserveInt64(1);
  Int64 n = values.size();
  for( Integer i=0; i<n; ++i )
    s->reserve(values[i]);
}

template<typename DataType>
void _directPut(ISerializer* s,Span<const DataType> values)
{
  s->putInt64(values.size());
  s->putSpan(values);
}

void _directPut(ISerializer* s,Span<const bool> values)
{
  Int64 n = values.size();
  s->putInt64(n);
  UniqueArray<Byte> bytes(n);
  for( Int64 i=0; i<n; ++i )
    bytes[i] = values[i] ? 1 : 0;

  s->putSpan(bytes);
}

void _directPut(ISerializer* s,Span<const String> values)
{
  Int64 n = values.size();
  s->putInt64(n);
  for( Integer i=0; i<n; ++i )
    s->put(values[i]);
}

template<typename DataType>
void _directGet(ISerializer* s,Array<DataType>& values)
{
  Int64 n = s->getInt64();
  //std::cout << "GET_N=" << n << '\n';
  values.resize(n);
  s->getSpan(values);
}

void _directGet(ISerializer* s,Array<bool>& values)
{
  Int64 n = s->getInt64();
  values.resize(n);

  UniqueArray<Byte> bytes(n);
  s->getSpan(bytes);

  for( Integer i=0; i<n; ++i )
    values[i] = (bytes[i]!=0);
}

void _directGet(ISerializer* s,Array<String>& values)
{
  Int64 n = s->getInt64();
  values.resize(n);

  for( Integer i=0; i<n; ++i )
    s->get(values[i]);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class ScalarPropertyType
: public IPropertyType
{
 public:

  ScalarPropertyType()
  {
    m_type_name = DataTypeTraitsT<DataType>::name();
  }

 public:

  void print(std::ostream& o,PropertyVariant* v) override
  {
    SmallVariant* x = v->getScalar();
    DataType d = DataType();
    x->value(d);
    o << d;
  }

  const String& typeName() const override
  {
    return m_type_name;
  }

  void serializeReserve(ISerializer* s,PropertyVariant* v) override
  {
    SmallVariant* x = v->getScalar();
    DataType d = DataType();
    x->value(d);
    _directReserveScalar(s,d);
  }

  void serializePut(ISerializer* s,PropertyVariant* v) override
  {
    SmallVariant* x = v->getScalar();
    DataType d = DataType();
    x->value(d);
    _directPutScalar(s,d);
  }

 private:
  String m_type_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class ArrayPropertyType
: public IPropertyType
{
 public:
  ArrayPropertyType()
  {
    m_type_name = String(DataTypeTraitsT<DataType>::name()) + "[]";
  }
 public:
  virtual void print(std::ostream& o,PropertyVariant* v)
  {
    UniqueArray<DataType>* x = v->get(DataType());
    Integer n = x->size();
    o << "(size=" << n;
    if (n>=1){
      for( Integer i=0; i<n; ++i )
        o << ',' << '[' << i << "]=" << x->operator[](i);
    }
    o << ')';
  }
  virtual const String& typeName() const
  {
    return m_type_name;
  }
  virtual void serializeReserve(ISerializer* s,PropertyVariant* v)
  {
    UniqueArray<DataType>* x = v->get(DataType());
    //s->reserve(DataTypeTraitsT<DataType>::type(),x->size());
    _directReserve(s,x->constSpan());
  }
  virtual void serializePut(ISerializer* s,PropertyVariant* v)
  {
    UniqueArray<DataType>* x = v->get(DataType());
    _directPut(s,x->constSpan());
  }
 private:
  String m_type_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * TODO: donner un peu plus d'informations pour les exceptions.
 * TODO: fusionner avec IData
 * TODO: faire un visiteur dessus (utiliser celui de IData)
 */
class PropertiesImpl
: public PropertiesImplBase
, public TraceAccessor
{
 public:

  // A modifier si les sérialisations sont modifiées et incompatibles avec les
  // anciennes versions. La version 2 (décembre 2019) utilise la sérialisation
  // des chaînes de caractères sur un Int64.
  static const Int32 SERIALIZE_VERSION = 2;

 public:

  typedef std::map<String,PropertyVariant*> MapType;

 public:

  PropertiesImpl(IPropertyMng* pm,const String& name);
  ~PropertiesImpl();

 public:

  virtual void deleteMe() { delete this; }

 public:
  
  IPropertyMng* m_property_mng;
  PropertiesImpl* m_parent_property;
  String m_name;
  String m_full_name;
  MapType m_property_map;
  UniqueArray<IPropertyType*> m_types;
  
 public:

  template<typename DataType>
  bool getScalarValue(const String& name,DataType& value)
  {
    typename MapType::const_iterator v = m_property_map.find(name);
    if (v==m_property_map.end()){
      return false;
    }

    SmallVariant* x = v->second->getScalar();
    if (!x)
      throw ArgumentException(A_FUNCINFO,"Bad data dimension for property (expecting scalar but property is array)");
    x->value(value);
    return true;
  }

  template<typename DataType>
  void _setScalarValue(SmallVariant& s,const DataType& value)
  {
    s.setValueAll(value);
  }
  void _setScalarValue(SmallVariant& s,const String& value)
  {
    s.setValue(value);
  }

  template<typename DataType>
  DataType setScalarValue(const String& name,const DataType& value)
  {
    DataType old_value = DataType();
    MapType::iterator v = m_property_map.find(name);
    if (v!=m_property_map.end()){
      SmallVariant* x = v->second->getScalar();
      if (!x)
        throw ArgumentException(A_FUNCINFO,"Bad data dimension for property (expecting scalar but property is array)");
      // Récupère l'ancienne valeur
      x->value(old_value);
      _setScalarValue(*x,value);
    }
    else{
      SmallVariant sv;
      _setScalarValue(sv,value);
      m_property_map.insert(std::make_pair(name,PropertyVariant::create(sv)));
    }
    return old_value;
  }

  template<typename DataType>
  void setArrayValue(const String& name,ConstArrayView<DataType> value)
  {
    MapType::iterator v = m_property_map.find(name);
    if (v!=m_property_map.end()){
      UniqueArray<DataType>* x = v->second->get(DataType());
      if (!x)
        throw ArgumentException(A_FUNCINFO,"Bad datatype for property");
      x->copy(value);
    }
    else{
      m_property_map.insert(std::make_pair(name,PropertyVariant::create(value)));
    }
  }

  template<typename DataType>
  void getArrayValue(const String& name,Array<DataType>& value)
  {
    MapType::const_iterator v = m_property_map.find(name);
    if (v==m_property_map.end()){
      value.clear();
      return;
    }
    UniqueArray<DataType>* x = v->second->get(DataType());
    if (!x)
      throw ArgumentException(A_FUNCINFO,"Bad datatype for property");
    value.copy(*x);
  }

 public:
  
  void print(std::ostream& o);

  void serialize(ISerializer* serializer);
  
  void serializeReserve(ISerializer* serializer);
  void serializePut(ISerializer* serializer);
  void serializeGet(ISerializer* serializer);
  
 private:
  
  template<typename DataType> void
  _serializeGetArray(ISerializer* serializer,const String& name,const DataType&)
  {
    UniqueArray<DataType> values;
    _directGet(serializer,values);
    setArrayValue(name,values.constView());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertiesImpl::
PropertiesImpl(IPropertyMng* pm,const String& name)
: TraceAccessor(pm->traceMng())
, m_property_mng(pm)
, m_parent_property(0)
, m_name(name)
, m_full_name(name)
{
  m_types.resize(PropertyVariant::NB_TYPE);
  m_types.fill(0);
  m_types[PropertyVariant::PV_ScalarReal] = new ScalarPropertyType<Real>();
  m_types[PropertyVariant::PV_ScalarInt32] = new ScalarPropertyType<Int32>();
  m_types[PropertyVariant::PV_ScalarInt64] = new ScalarPropertyType<Int64>();
  m_types[PropertyVariant::PV_ScalarBool] = new ScalarPropertyType<bool>();
  m_types[PropertyVariant::PV_ScalarString] = new ScalarPropertyType<String>();

  m_types[PropertyVariant::PV_ArrayReal] = new ArrayPropertyType<Real>();
  m_types[PropertyVariant::PV_ArrayInt32] = new ArrayPropertyType<Int32>();
  m_types[PropertyVariant::PV_ArrayInt64] = new ArrayPropertyType<Int64>();
  m_types[PropertyVariant::PV_ArrayBool] = new ArrayPropertyType<bool>();
  m_types[PropertyVariant::PV_ArrayString] = new ArrayPropertyType<String>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertiesImpl::
~PropertiesImpl()
{
  MapType::iterator v = m_property_map.begin();
  MapType::iterator vend = m_property_map.end();
  for( ; v!=vend; ++v )
    delete v->second;

  for( Integer i=0, n=m_types.size(); i<n; ++i )
    delete m_types[i];

  info(5) << "DESTROY PROPERTY name=" << m_name << " this=" << this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertiesImpl::
print(std::ostream& o)
{
  MapType::iterator v = m_property_map.begin();
  MapType::iterator vend = m_property_map.end();
  for( ; v!=vend; ++v ){
    PropertyVariant* p = v->second;
    PropertyVariant::eType et = p->type();
    IPropertyType* pt = m_types[et];
    o << "  " << v->first << " = ";
    if (pt){
      o << "(" << pt->typeName();
      o << ") ";
      pt->print(o,p);
    }
    else{
      o << "??";
    }
    o << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertiesImpl::
serialize(ISerializer* serializer)
{
  switch(serializer->mode()){
  case ISerializer::ModeReserve:
    serializeReserve(serializer);
    break;
  case ISerializer::ModePut:
    serializePut(serializer);
    break;
  case ISerializer::ModeGet:
    serializeGet(serializer);
    break;
  }
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertiesImpl::
serializeReserve(ISerializer* serializer)
{
  serializer->reserveInt32(1); // SERIALIZE_VERSION
  serializer->reserveInt64(1); // Nombre d'éléments dans la map

  MapType::iterator v = m_property_map.begin();
  MapType::iterator vend = m_property_map.end();
  for( ; v!=vend; ++v ){
    PropertyVariant* p = v->second;
    PropertyVariant::eType et = p->type();
    serializer->reserveInt32(1);
    serializer->reserve(v->first);
    IPropertyType* pt = m_types[et];
    pt->serializeReserve(serializer,p);
  }
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertiesImpl::
serializePut(ISerializer* serializer)
{
  serializer->putInt32(SERIALIZE_VERSION);
    
  Int64 n = (Int64)m_property_map.size();
  serializer->putInt64(n);

  MapType::iterator v = m_property_map.begin();
  MapType::iterator vend = m_property_map.end();
  for( ; v!=vend; ++v ){
    PropertyVariant* p = v->second;
    PropertyVariant::eType et = p->type();
    IPropertyType* pt = m_types[et];
    serializer->putInt32(et);
    serializer->put(v->first);
    pt->serializePut(serializer,p);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertiesImpl::
serializeGet(ISerializer* serializer)
{
  Int64 version = serializer->getInt32();
  if (version!=SERIALIZE_VERSION){
    // La relecture se fait avec une protection issue d'une ancienne version de Arcane
    // et qui n'est pas compatible. Affiche un avertissement et ne fait rien.
    // (NOTE: cela risque quand même de poser problême pour ceux qui utilisent les
    // propriétés donc il faudrait peut-être faire un fatal ?)
    pwarning() << "Can not reading properties from imcompatible checkpoint";
    return;
  }

  Int64 n = serializer->getInt64();
  String name;
  for( Integer i=0; i<n; ++i ){
    Int32 type = serializer->getInt32();
    serializer->get(name);
    //std::cout << "TYPE=" << type << " name=" << name << '\n';
    //TODO: Verifier validités des valeurs + faire ca proprement sans switch
    switch(type){
    case PropertyVariant::PV_ScalarReal: setScalarValue(name,serializer->getReal()); break;
    case PropertyVariant::PV_ScalarInt32:setScalarValue(name,serializer->getInt32()); break;
    case PropertyVariant::PV_ScalarInt64:setScalarValue(name,serializer->getInt64()); break;
    case PropertyVariant::PV_ScalarBool: setScalarValue(name,(bool)serializer->getByte()); break;
    case PropertyVariant::PV_ScalarString: { String str; serializer->get(str); setScalarValue(name,str); } break;

    case PropertyVariant::PV_ArrayReal: _serializeGetArray(serializer,name,Real()); break;
    case PropertyVariant::PV_ArrayInt32:_serializeGetArray(serializer,name,Int32()); break;
    case PropertyVariant::PV_ArrayInt64:_serializeGetArray(serializer,name,Int64()); break;
    case PropertyVariant::PV_ArrayBool:_serializeGetArray(serializer,name,bool()); break;
    case PropertyVariant::PV_ArrayString:_serializeGetArray(serializer,name,String()); break;
    default:
      throw FatalErrorException(A_FUNCINFO,"Bad type");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Properties::
Properties(IPropertyMng* pm,const String& aname)
: m_p(0)
, m_ref(0)
{
  PropertiesImpl* pi = pm->getPropertiesImpl(aname);
  bool has_create = false;
  if (!pi){
    pi = new PropertiesImpl(pm,aname);
    has_create = true;
  }
  m_p = pi;
  m_ref = pi;
  if (has_create)
    pm->registerProperties(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Properties::
Properties(const Properties& parent_property,const String& aname)
{
  String full_name = parent_property.fullName()+"."+aname;
  IPropertyMng* pm = parent_property.propertyMng();
  PropertiesImpl* pi = pm->getPropertiesImpl(full_name);

  bool has_create = false;
  if (!pi){
    pi = new PropertiesImpl(pm,full_name);
    has_create = true;
  }
  m_p = pi;
  m_ref = pi;
  if (has_create)
    pm->registerProperties(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Properties::
Properties(PropertiesImpl* p)
: m_p(p)
, m_ref(m_p)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Properties::
~Properties()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Properties::
Properties(const Properties& rhs)
: m_p(rhs.m_p)
, m_ref(rhs.m_ref)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Properties& Properties::
operator=(const Properties& rhs)
{
  if (&rhs!=this){
    m_p = rhs.m_p;
    m_ref = rhs.m_ref;
  }
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
setBool(const String& aname,bool value)
{   
  m_p->setScalarValue(aname,value);
}
void Properties::
set(const String& aname,bool value)
{   
  setBool(aname,value);
}
bool Properties::
getBoolWithDefault(const String& aname,bool default_value) const
{
  bool v = default_value;
  get(aname,v);
  return v;
}
bool Properties::
getBool(const String& aname) const
{
  return getBoolWithDefault(aname,false);
}
bool Properties::
get(const String& aname,bool& value) const
{
  return m_p->getScalarValue(aname,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
setInt32(const String& aname,Int32 value)
{   
  m_p->setScalarValue(aname,value);
}
void Properties::
set(const String& aname,Int32 value)
{   
  setInt32(aname,value);
}
Int32 Properties::
getInt32WithDefault(const String& name,Int32 default_value) const
{
  Int32 v = default_value;
  m_p->getScalarValue(name,v);
  return v;
}
Int32 Properties::
getInt32(const String& name) const
{
  return getInt32WithDefault(name,0);
}
bool Properties::
get(const String& name,Int32& value) const
{
  return m_p->getScalarValue(name,value);
  //value = getInt32(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
setInt64(const String& aname,Int64 value)
{   
  m_p->setScalarValue(aname,value);
}
void Properties::
set(const String& aname,Int64 value)
{   
  setInt64(aname,value);
}
Int64 Properties::
getInt64WithDefault(const String& aname,Int64 default_value) const
{
  Int64 v = default_value;
  m_p->getScalarValue(aname,v);
  return v;
}
Int64 Properties::
getInt64(const String& aname) const
{
  return getInt64WithDefault(aname,0);
}
bool Properties::
get(const String& aname,Int64& value) const
{
  return m_p->getScalarValue(aname,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
setInteger(const String& aname,Integer value)
{
  set(aname,value);
}
Integer Properties::
getIntegerWithDefault(const String& aname,Integer default_value) const
{
  Integer x = default_value;
  m_p->getScalarValue(aname,x);
  return x;
}
Integer Properties::
getInteger(const String& name) const
{
  return getIntegerWithDefault(name,0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
setReal(const String& aname,Real value)
{   
  m_p->setScalarValue(aname,value);
}
void Properties::
set(const String& aname,Real value)
{   
  setReal(aname,value);
}
Real Properties::
getRealWithDefault(const String& aname,Real default_value) const
{
  Real v = default_value;
  m_p->getScalarValue(aname,v);
  return v;
}
Real Properties::
getReal(const String& aname) const
{
  return getRealWithDefault(aname,0.0);
}
bool Properties::
get(const String& aname,Real& value) const
{
  return m_p->getScalarValue(aname,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
setString(const String& aname,const String& value)
{
  m_p->setScalarValue(aname,value);
}
void Properties::
set(const String& aname,const String& value)
{
  setString(aname,value);
}
String Properties::
getStringWithDefault(const String& aname,const String& default_value) const
{
  String v = default_value;
  m_p->getScalarValue(aname,v);
  return v;
}
String Properties::
getString(const String& aname) const
{
  return getStringWithDefault(aname,String());
}
bool Properties::
get(const String& aname,String& value) const
{
  return m_p->getScalarValue(aname,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
set(const String& aname,BoolConstArrayView value)
{
  m_p->setArrayValue(aname,value);
}
void Properties::
get(const String& aname,BoolArray& value) const
{
  m_p->getArrayValue(aname,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
set(const String& aname,Int32ConstArrayView value)
{
  m_p->setArrayValue(aname,value);
} 
void Properties::
get(const String& aname,Int32Array& value) const
{
  m_p->getArrayValue(aname,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
set(const String& aname,Int64ConstArrayView value)
{
  m_p->setArrayValue(aname,value);
} 
void Properties::
get(const String& aname,Int64Array& value) const
{
  m_p->getArrayValue(aname,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
set(const String& aname,RealConstArrayView value)
{
  m_p->setArrayValue(aname,value);
} 
void Properties::
get(const String& aname,RealArray& value) const
{
  m_p->getArrayValue(aname,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
set(const String& aname,StringConstArrayView value)
{
  m_p->setArrayValue(aname,value);
}
void Properties::
get(const String& aname,StringArray& value) const
{
  m_p->getArrayValue(aname,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
print(std::ostream& o) const
{
  m_p->print(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
serialize(ISerializer* serializer)
{
  m_p->serialize(serializer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& Properties::
name() const
{
  return m_p->m_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& Properties::
fullName() const
{
  return m_p->m_full_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPropertyMng* Properties::
propertyMng() const
{
  return m_p->m_property_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Properties::
destroy()
{
  m_p->m_property_mng->destroyProperties(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
