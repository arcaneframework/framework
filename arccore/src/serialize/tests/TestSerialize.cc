// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/serialize/BasicSerializer.h"

#include "arccore/base/Ref.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/ValueFiller.h"
#include "arccore/base/BasicDataType.h"
#include "arccore/base/Float128.h"
#include "arccore/base/Int128.h"

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISerializeValue
{
 public:

  virtual ~ISerializeValue() = default;

  virtual void serialize(ISerializer* s) = 0;
  virtual void checkValid() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
using Float64 = Real;
template <typename T> class ValueTraits;
#define VALUE_TRAITS(type_name,basic_type_name) \
  template <> class ValueTraits<type_name> \
  { \
   public: \
\
    static eBasicDataType dataType() { return eBasicDataType::basic_type_name; } \
    static void getValue(ISerializer* s, type_name& v) { v = s->get##type_name(); } \
  };

#define VALUE_TRAITS2(type_name) VALUE_TRAITS(type_name,type_name)

VALUE_TRAITS2(Byte);
VALUE_TRAITS2(Int8);
VALUE_TRAITS2(Int16);
VALUE_TRAITS2(Int32)
VALUE_TRAITS2(Int64);
VALUE_TRAITS2(Float16);
VALUE_TRAITS2(Float32);
VALUE_TRAITS2(BFloat16);
VALUE_TRAITS(Real,Float64);
VALUE_TRAITS2(Float128);
VALUE_TRAITS2(Int128);

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class SerializeValue
: public ISerializeValue
{
  using ValueTraitsType = ValueTraits<DataType>;

 public:

  SerializeValue()
  : m_data_type(ValueTraitsType::dataType())
  {}

 public:

  void serialize(ISerializer* s) override
  {
    Int64 size = m_array_values.size();
    switch (s->mode()) {
    case ISerializer::ModeReserve:
      std::cout << "ReserveArray type=" << m_data_type << " size=" << m_array_values.size() << "\n";
      s->reserveArray(m_array_values);
      if (size > 0)
        s->reserve(m_data_type, 1);
      break;
    case ISerializer::ModePut:
      std::cout << "PutArray type=" << m_data_type << " size=" << m_array_values.size() << "\n";
      s->putArray(m_array_values);
      if (size > 0)
        s->put(m_array_values[0]);
      break;
    case ISerializer::ModeGet:
      s->getArray(m_result_array_values);
      if (size > 0)
        ValueTraitsType::getValue(s, m_unique_value);
    }
  }

  void checkValid() override
  {
    std::cout << "ref_size=" << m_array_values.size()
              << " result_size=" << m_result_array_values.size() << "\n";
    ASSERT_EQ(m_array_values, m_result_array_values);
    ASSERT_EQ(m_array_values[0], m_unique_value);
  }

  void resizeAndFill(Int32 size)
  {
    m_array_values.resize(size);
    ValueFiller::fillRandom(542, m_array_values.span());
  }

 public:

  UniqueArray<DataType> m_array_values;
  UniqueArray<DataType> m_result_array_values;
  DataType m_unique_value = {};
  eBasicDataType m_data_type = eBasicDataType::Unknown;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StringSerializeValue
: public ISerializeValue
{
 public:

  explicit StringSerializeValue(const String& v)
  : m_ref_string(v)
  {}

 public:

  void serialize(ISerializer* s) override
  {
    switch (s->mode()) {
    case ISerializer::ModeReserve:
      s->reserve(m_ref_string);
      break;
    case ISerializer::ModePut:
      s->put(m_ref_string);
      break;
    case ISerializer::ModeGet:
      s->get(m_result_string);
    }
  }

  void checkValid() override
  {
    ASSERT_EQ(m_ref_string, m_result_string);
  }

 public:

  String m_ref_string;
  String m_result_string;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SerializeValueList
{
 public:

  ~SerializeValueList()
  {
    for (ISerializeValue* v : m_values)
      delete v;
  }
  void doSerialize(ISerializer* s)
  {
    for (ISerializeValue* v : m_values)
      v->serialize(s);
  }
  void checkValid()
  {
    for (ISerializeValue* v : m_values)
      v->checkValid();
  }

  template <typename DataType> void add(Int32 size)
  {
    auto* sval = new SerializeValue<DataType>();
    sval->resizeAndFill(size);
    m_values.add(sval);
  }
  void addString(const String& v)
  {
    m_values.add(new StringSerializeValue(v));
  }

 public:

  UniqueArray<ISerializeValue*> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void _doMisc()
{
  Ref<ISerializer> serializer_ref = createSerializer();
  ISerializer* serializer = serializer_ref.get();

  SerializeValueList values;

  values.add<Float16>(12679);
  values.add<BFloat16>(3212);
  values.add<Int8>(6357);
  values.add<Float32>(983);
  values.add<Int16>(16353);
  values.add<Real>(29123);
  values.add<Int32>(16);
  values.add<Byte>(3289);
  values.add<Int64>(12932);
  values.add<Float128>(19328);
  values.add<Int128>(32422);
  values.addString("Ceci est un test de chaîne de caractères");

  serializer->setMode(ISerializer::ModeReserve);
  values.doSerialize(serializer);
  serializer->allocateBuffer();
  serializer->setMode(ISerializer::ModePut);
  values.doSerialize(serializer);
  serializer->setMode(ISerializer::ModeGet);
  values.doSerialize(serializer);

  values.checkValid();
}

TEST(Serialize, Misc)
{
  try {
    _doMisc();
  }
  catch (const Exception& e) {
    e.write(std::cerr);
    throw;
  }
}
