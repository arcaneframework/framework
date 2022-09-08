// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UdunitsUnitSystem.h                                         (C) 2000-2014 */
/*                                                                           */
/* Gestion de système d'unité physique utilisant 'udunits2'.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/FactoryService.h"
#include "arcane/AbstractService.h"
#include "arcane/IPhysicalUnitSystemService.h"
#include "arcane/IPhysicalUnitSystem.h"
#include "arcane/IPhysicalUnitConverter.h"
#include "arcane/IPhysicalUnit.h"

#include <udunits2.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

class UdunitsPhysicalUnitConverter;
class UdunitsPhysicalUnit;
class UdunitsPhysicalUnitSystem;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UdunitsPhysicalUnit
: public IPhysicalUnit
{
 public:

  friend class UdunitsPhysicalUnitSystem;

 public:

  UdunitsPhysicalUnit(UdunitsPhysicalUnitSystem*,::ut_unit* unit,
                      const String& name)
  : m_unit(unit)
  {
    // Pour garantir que la chaîne est allouée dynamiquement
    m_name = String::fromUtf8(name.utf8());
  }

  UdunitsPhysicalUnit(::ut_unit* unit)
  : m_unit(unit)
  {
  }

  virtual ~UdunitsPhysicalUnit()
  {
    if (m_unit)
      ut_free(m_unit);
  }

  virtual const String& name() const
  {
    return m_name;
  }

 private:

  ::ut_unit* m_unit;
  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UdunitsPhysicalUnitConverter
: public IPhysicalUnitConverter
{
 public:

  UdunitsPhysicalUnitConverter(UdunitsPhysicalUnitSystem* unit_system,
                               UdunitsPhysicalUnit* from_unit,
                               UdunitsPhysicalUnit* to_unit,
                               ::cv_converter* converter)
  : m_unit_system(unit_system), m_from_unit(from_unit),
    m_to_unit(to_unit), m_is_from_owned(true), m_is_to_owned(true), m_converter(converter)
  {
  }

  ~UdunitsPhysicalUnitConverter()
  {
    if (m_is_from_owned)
      delete m_from_unit;
    if (m_is_to_owned)
      delete m_to_unit;
  }

 public:

  virtual Real convert(Real value)
  {
    double new_value = cv_convert_double(m_converter,value);
    return new_value;
  }

  virtual void convert(RealConstArrayView input_values,
                       RealArrayView output_values)
  {
    Integer nb = input_values.size();
    Integer nb_out = output_values.size();
    if (nb!=nb_out)
      throw ArgumentException(A_FUNCINFO,"input and ouput arrays do not have the same number of elements");
    cv_convert_doubles(m_converter,input_values.data(),nb,output_values.data());
  }

  virtual IPhysicalUnit* fromUnit()
  {
    return m_from_unit;
  }

  virtual IPhysicalUnit* toUnit()
  {
    return m_to_unit;
  }

 public:

 private:
  UdunitsPhysicalUnitSystem* m_unit_system;
  UdunitsPhysicalUnit* m_from_unit;
  UdunitsPhysicalUnit* m_to_unit;
  bool m_is_from_owned;
  bool m_is_to_owned;
  ::cv_converter* m_converter;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UdunitsPhysicalUnitSystem
: public TraceAccessor
, public IPhysicalUnitSystem
{
 public:

  UdunitsPhysicalUnitSystem(ITraceMng* tm)
  : TraceAccessor(tm), m_unit_system(0)
  {
  }

  virtual ~UdunitsPhysicalUnitSystem()
  {
    if (m_unit_system)
      ut_free_system(m_unit_system);
  }

  virtual IPhysicalUnitConverter* createConverter(IPhysicalUnit* from,IPhysicalUnit* to)
  {
    // Normalement déjà créé.
    _checkCreateUnitSystem();
    UdunitsPhysicalUnit* from_unit = dynamic_cast<UdunitsPhysicalUnit*>(from);
    UdunitsPhysicalUnit* to_unit = dynamic_cast<UdunitsPhysicalUnit*>(to);
    if (!from_unit || !to_unit)
      throw ArgumentException(A_FUNCINFO,"can not convert units to this system unit");
    return _createConverter(from_unit,to_unit);
  }

  virtual IPhysicalUnitConverter* createConverter(const String& from,const String& to)
  {
    info() << "Create unit converter from='" << from << "' to='" << to << "'";
    _checkCreateUnitSystem();
    UdunitsPhysicalUnit* from_unit = _createUnit(from);
    UdunitsPhysicalUnit* to_unit = _createUnit(to);
    //TODO: detruire les unites en cas d'erreur.
    return _createConverter(from_unit,to_unit);
  }

 private:
  
  UdunitsPhysicalUnitConverter* _createConverter(UdunitsPhysicalUnit* from_unit,UdunitsPhysicalUnit* to_unit)
  {
    ::cv_converter* cv_cvt = ::ut_get_converter(from_unit->m_unit,to_unit->m_unit);
    if (!cv_cvt)
      throw FatalErrorException(A_FUNCINFO,
                                String::format("Can not convert from '{0}' to '{1}' because units are not convertible",
                                               from_unit->name(),to_unit->name()));
    UdunitsPhysicalUnitConverter* cvt = new UdunitsPhysicalUnitConverter(this,from_unit,to_unit,cv_cvt);
    return cvt;
  }

  void _checkCreateUnitSystem()
  {
    //TODO Ajouter lock...
    if (!m_unit_system)
      m_unit_system = ut_read_xml(0);
    if (!m_unit_system)
      throw FatalErrorException(A_FUNCINFO,"Can not load unit system");
  }

  UdunitsPhysicalUnit* _createUnit(const String& name)
  {
    ::ut_unit* unit = ut_parse(m_unit_system,(const char*)name.utf8().data(),UT_UTF8);
    if (!unit)
      throw FatalErrorException(A_FUNCINFO,String::format("Can not create unit from string '{0}'",name));
    return new UdunitsPhysicalUnit(this,unit,name);
  }

 public:
  

 private:

  ::ut_system* m_unit_system;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de gestion de système d'unité physique utilisant 'udunits2'.
 */
class UdunitsUnitSystemService
: public AbstractService
, public IPhysicalUnitSystemService
{
 public:

  UdunitsUnitSystemService(const ServiceBuildInfo& sbi);
  virtual ~UdunitsUnitSystemService() {}

 public:

  virtual void build();

 public:

  virtual IPhysicalUnitSystem* createStandardUnitSystem()
  {
    UdunitsPhysicalUnitSystem* s = new UdunitsPhysicalUnitSystem(traceMng());
    return s;
  }

 private:

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UdunitsUnitSystemService::
UdunitsUnitSystemService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UdunitsUnitSystemService::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(UdunitsUnitSystemService,
                                    IPhysicalUnitSystemService,
                                    Udunits);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
