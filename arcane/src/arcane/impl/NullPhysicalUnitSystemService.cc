// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NullPhysicalUnitSystemService.h                             (C) 2000-2020 */
/*                                                                           */
/* Gestion de système d'unité physique par défaut.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ArrayView.h"

#include "arcane/IPhysicalUnitSystemService.h"
#include "arcane/IPhysicalUnitSystem.h"
#include "arcane/IPhysicalUnitConverter.h"
#include "arcane/IPhysicalUnit.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullPhysicalUnitConverter;
class NullPhysicalUnit;
class NullPhysicalUnitSystem;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullPhysicalUnit
: public IPhysicalUnit
{
 public:

  friend class NullPhysicalUnitSystem;

 public:

  const String& name() const override
  {
    return m_name;
  }

 private:

  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullPhysicalUnitConverter
: public IPhysicalUnitConverter
{
 public:

  NullPhysicalUnitConverter()
  {
  }

 public:

  Real convert(Real value) override
  {
    ARCANE_UNUSED(value);
    throw NotSupportedException(A_FUNCINFO);
  }

  void convert(RealConstArrayView input_values,
               RealArrayView output_values) override
  {
    ARCANE_UNUSED(input_values);
    ARCANE_UNUSED(output_values);
    throw NotSupportedException(A_FUNCINFO);
  }

  IPhysicalUnit* fromUnit() override
  {
    return m_from_unit;
  }

  IPhysicalUnit* toUnit() override
  {
    return m_to_unit;
  }

 public:

 private:
  NullPhysicalUnit* m_from_unit;
  NullPhysicalUnit* m_to_unit;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullPhysicalUnitSystem
: public IPhysicalUnitSystem
{
 public:

  NullPhysicalUnitSystem() = default;

  IPhysicalUnitConverter* createConverter(IPhysicalUnit* from,IPhysicalUnit* to) override
  {
    ARCANE_UNUSED(from);
    ARCANE_UNUSED(to);
    NullPhysicalUnitConverter* cvt = new NullPhysicalUnitConverter();
    return cvt;
  }

  IPhysicalUnitConverter* createConverter(const String& from,const String& to) override
  {
    ARCANE_UNUSED(from);
    ARCANE_UNUSED(to);
    NullPhysicalUnitConverter* cvt = new NullPhysicalUnitConverter();
    return cvt;
  }

 public:
  
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NullPhysicalUnitSystemService
: public IPhysicalUnitSystemService
{
 public:

  NullPhysicalUnitSystemService() = default;

 public:

  void build() override {}

 public:

  IPhysicalUnitSystem* createStandardUnitSystem() override
  {
    NullPhysicalUnitSystem* s = new NullPhysicalUnitSystem();
    return s;
  }

 private:

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IPhysicalUnitSystemService*
createNullPhysicalUnitSystemService()
{
  IPhysicalUnitSystemService* s = new NullPhysicalUnitSystemService();
  s->build();
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
