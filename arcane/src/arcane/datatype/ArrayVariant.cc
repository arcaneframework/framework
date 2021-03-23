// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayVariant.cc                                             (C) 2000-2006 */
/*                                                                           */
/* Type de base polymorphe pour les tableaux mono-dim (dimension 1).         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"

#include "arcane/datatype/ArrayVariant.h"
#include "arcane/datatype/BadVariantTypeException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
ArrayVariant(eType type,Integer asize)
: VariantBase(1, type)
, m_allocated_array(0)
{
  switch(type){
  case TReal:
    {
      RealArray* v = new RealUniqueArray(asize);
      m_allocated_array = v;
      m_real_value = *v;
    }
    break;
  case TInt64:
    {
      Int64Array* v = new Int64UniqueArray(asize);
      m_allocated_array = v;
      m_int64_value = *v;
    }
    break;
  case TInt32:
    {
      Int32Array* v = new Int32UniqueArray(asize);
      m_allocated_array = v;
      m_int32_value = *v;
    }
    break;
  case TBool:
    {
      BoolArray* v = new BoolUniqueArray(asize);
      m_allocated_array = v;
      m_bool_value = *v;
    }
    break;
  case TString:
    {
      StringArray* v = new StringUniqueArray(asize);
      m_allocated_array = v;
      m_string_value = *v;
    }
    break;
  case TReal2:
    {
      Real2Array* v = new Real2UniqueArray(asize);
      m_allocated_array = v;
      m_real2_value = *v;
    }
    break;
  case TReal3:
    {
      Real3Array* v = new Real3UniqueArray(asize);
      m_allocated_array = v;
      m_real3_value = *v;
    }
    break;
  case TReal2x2:
    {
      Real2x2Array* v = new Real2x2UniqueArray(asize);
      m_allocated_array = v;
      m_real2x2_value = *v;
    }
    break;
  case TReal3x3:
    {
      Real3x3Array* v = new Real3x3UniqueArray(asize);
      m_allocated_array = v;
      m_real3x3_value = *v;
    }
    break;
  default:
    throw BadVariantTypeException("ArrayVariant::ArrayVariant(eType,Integer)",type);
  }
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
ArrayVariant(ArrayView<Real> data)
: VariantBase(1,TReal)
, m_real_value(data)
, m_allocated_array(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
ArrayVariant(ArrayView<Real2> data)
: VariantBase(1,TReal2)
, m_real2_value(data)
, m_allocated_array(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
ArrayVariant(ArrayView<Real3> data)
: VariantBase(1, TReal3)
, m_real3_value(data)
, m_allocated_array(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
ArrayVariant(ArrayView<Real2x2> data)
: VariantBase(1,TReal2x2)
, m_real2x2_value(data)
, m_allocated_array(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
ArrayVariant(ArrayView<Real3x3> data)
: VariantBase(1,TReal3x3)
, m_real3x3_value(data)
, m_allocated_array(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
ArrayVariant(ArrayView<Int32> data)
: VariantBase(1,TInt32)
, m_int32_value(data)
, m_allocated_array(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
ArrayVariant(ArrayView<Int64> data)
: VariantBase(1,TInt64)
, m_int64_value(data)
, m_allocated_array(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
ArrayVariant(ArrayView<bool> data)
: VariantBase(1,TBool)
, m_bool_value(data)
, m_allocated_array(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
ArrayVariant(ArrayView<String> data)
: VariantBase(1,TString)
, m_string_value(data)
, m_allocated_array(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayVariant::
~ArrayVariant()
{
  _destroy();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayVariant::
_destroy()
{
  if (!m_allocated_array)
    return;
  switch(m_type){
  case TReal: delete reinterpret_cast<RealArray*>(m_allocated_array); break;
  case TReal2: delete reinterpret_cast<Real2Array*>(m_allocated_array); break;
  case TReal3: delete reinterpret_cast<Real3Array*>(m_allocated_array); break;
  case TReal2x2: delete reinterpret_cast<Real2x2Array*>(m_allocated_array); break;
  case TReal3x3: delete reinterpret_cast<Real3x3Array*>(m_allocated_array); break;
  case TInt64: delete reinterpret_cast<Int64Array*>(m_allocated_array); break;
  case TInt32: delete reinterpret_cast<Int32Array*>(m_allocated_array); break;
  case TBool: delete reinterpret_cast<BoolArray*>(m_allocated_array); break;
  case TString: delete reinterpret_cast<StringArray*>(m_allocated_array); break;
  default:
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ArrayVariant::
size() const
{
  switch(m_type){
  case TReal:
    return m_real_value.size();
  case TReal2:
    return m_real2_value.size();
  case TReal3:
    return m_real3_value.size();
  case TReal2x2:
    return m_real2x2_value.size();
  case TReal3x3:
    return m_real3x3_value.size();
  case TBool:
    return m_bool_value.size();
  case TString:
    return m_string_value.size();
  case TInt32:
    return m_int32_value.size();
  case TInt64:
    return m_int64_value.size();
  default:
    break;
  }
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_DATATYPE_EXPORT std::ostream& 
operator<<(std::ostream& s, const ArrayVariant& x)
{
  s << "ArrayVariant [t="
    << x.typeName();
  
  s << ", adr=";
  Integer size = 0;
  switch (x.type())
  {
  case VariantBase::TReal:
    s << x.asReal().data();
    s << "], v=[ ";
    size = x.asReal().size();
    for (Integer i=0 ; i<size ; ++i)
      s << x.asReal()[i] << " ";
    break;
  case VariantBase::TReal2:
    s << x.asReal2().data();
    s << "], v=[ ";
    size = x.asReal2().size();
    for (Integer i=0 ; i<size ; ++i)
      s << x.asReal2()[i] << " ";
    break;
  case VariantBase::TReal3:
    s << x.asReal3().data();
    s << "], v=[ ";
    size = x.asReal3().size();
    for (Integer i=0 ; i<size ; ++i)
      s << x.asReal3()[i] << " ";
    break;
  case VariantBase::TReal2x2:
    s << x.asReal2().data();
    s << "], v=[ ";
    size = x.asReal2().size();
    for (Integer i=0 ; i<size ; ++i)
      s << x.asReal2x2()[i] << " ";
    break;
  case VariantBase::TReal3x3:
    s << x.asReal3x3().data();
    s << "], v=[ ";
    size = x.asReal3x3().size();
    for (Integer i=0 ; i<size ; ++i)
      s << x.asReal3x3()[i] << " ";
    break;
  case VariantBase::TInt64:
    s << x.asInt64().data();
    s << "], v=[ ";
    size = x.asInt64().size();
    for(Integer i=0 ; i<size ; ++i)
      s << x.asInt64()[i] << " ";
    break;
  case VariantBase::TInt32:
    s << x.asInteger().data();
    s << "], v=[ ";
    size = x.asInteger().size();
    for (Integer i=0 ; i<size ; ++i)
      s << x.asInteger()[i] << " ";
    break;
  case VariantBase::TBool:
    s << x.asBool().data();
    s << "], v=[ ";
    size = x.asBool().size();
    for (Integer i=0 ; i<size ; ++i)
      s << x.asBool()[i] << " ";
    break;
  case VariantBase::TString:
    s << x.asString().data();
    s << "], v=[ ";
    size = x.asString().size();
    for (Integer i=0 ; i<size ; ++i)
      s << x.asString()[i] << " ";
    break;
  default:
    break;
  }
  s << "]";
  
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IntegerArrayView ArrayVariant::
asInteger() 
{
#ifdef ARCANE_64BIT
  return m_int64_value;
#else
  return m_int32_value;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IntegerConstArrayView ArrayVariant::
asInteger() const
{
#ifdef ARCANE_64BIT
  return m_int64_value;
#else
  return m_int32_value;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
