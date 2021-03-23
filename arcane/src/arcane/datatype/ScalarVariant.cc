// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ScalarVariant.cc                                            (C) 2000-2005 */
/*                                                                           */
/* Type de base polymorphe pour les scalaires (dimension 0).                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/datatype/ScalarVariant.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant()
: VariantBase(0, TUnknown)
, m_real_value(0.)
, m_int32_value(0)
, m_int64_value(0)
, m_bool_value(false) 
{
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant(const ScalarVariant& v)
: VariantBase(0, v.m_type)
, m_real_value(v.m_real_value)
, m_real2_value(v.m_real2_value)
, m_real3_value(v.m_real3_value)
, m_real2x2_value(v.m_real2x2_value)
, m_real3x3_value(v.m_real3x3_value)
, m_int32_value(v.m_int32_value)
, m_int64_value(v.m_int64_value)
, m_bool_value(v.m_bool_value)
, m_string_value(v.m_string_value) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant(Real v)
: VariantBase(0, TReal)
, m_real_value(v)
, m_int32_value(0)
, m_int64_value(0)
, m_bool_value(false) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant(Real2 v)
: VariantBase(0, TReal2)
, m_real_value(0.)
, m_real2_value(v)
, m_int32_value(0)
, m_int64_value(0)
, m_bool_value(false) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant(Real3 v)
: VariantBase(0, TReal3)
, m_real_value(0.)
, m_real3_value(v)
, m_int32_value(0)
, m_int64_value(0)
, m_bool_value(false) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant(Real2x2 v)
: VariantBase(0, TReal2x2)
, m_real_value(0.)
, m_real2x2_value(v)
, m_int32_value(0)
, m_int64_value(0)
, m_bool_value(false) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant(Real3x3 v)
: VariantBase(0, TReal3x3)
, m_real_value(0.)
, m_real3x3_value(v)
, m_int32_value(0)
, m_int64_value(0)
, m_bool_value(false) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant(Int32 v)
: VariantBase(0, TInt32)
, m_real_value(0.)
, m_real2_value(Real2::null())
, m_real3_value(Real3::null())
, m_int32_value(v)
, m_int64_value(0)
, m_bool_value(false) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant(Int64 v)
: VariantBase(0, TInt64)
, m_real_value(0.)
, m_int32_value(0)
, m_int64_value(v)
, m_bool_value(false) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant(bool v)
: VariantBase(0, TBool)
, m_real_value(0.)
, m_int32_value(0)
, m_int64_value(0)
, m_bool_value(v) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ScalarVariant::
ScalarVariant(const String& v)
: VariantBase(0,TString)
, m_real_value(0.)
, m_int32_value(0)
, m_int64_value(0)
, m_bool_value(false)
, m_string_value(v) 
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ScalarVariant& ScalarVariant::
operator= (ScalarVariant v)
{ 
  m_dim = 0;
  m_type = v.m_type;
  m_real_value = v.m_real_value; 
  m_real2_value = v.m_real2_value; 
  m_real3_value = v.m_real3_value; 
  m_real2x2_value = v.m_real2x2_value; 
  m_real3x3_value = v.m_real3x3_value; 
  m_int64_value = v.m_int64_value; 
  m_int32_value = v.m_int32_value; 
  m_bool_value = v.m_bool_value; 
  m_string_value = v.m_string_value; 
  return (*this); 
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ScalarVariant::
asInteger() const
{
#ifdef ARCANE_64BIT
  return asInt64();
#else
  return asInt32();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
