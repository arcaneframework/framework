// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/serialize/BasicSerializer.h"

#include "arccore/base/Ref.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/ValueFiller.h"

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
template <typename T> class ValueTraits;
#define VALUE_TRAITS(type_name) \
  template <> class ValueTraits<type_name> \
  { \
   public: \
\
    static ISerializer::eDataType dataType() { return ISerializer::DT_##type_name; } \
    static void getValue(ISerializer* s, type_name& v) { v = s->get##type_name(); } \
  };

VALUE_TRAITS(Byte);
VALUE_TRAITS(Int8);
VALUE_TRAITS(Int16);
VALUE_TRAITS(Int32)
VALUE_TRAITS(Int64);
VALUE_TRAITS(Float16);
VALUE_TRAITS(Float32);
VALUE_TRAITS(BFloat16);
VALUE_TRAITS(Real);

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
      s->reserveArray(m_array_values);
      if (size > 0)
        s->reserve(m_data_type, 1);
      break;
    case ISerializer::ModePut:
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
  ISerializer::eDataType m_data_type;
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
