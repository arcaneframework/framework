// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArccoreApplicationBuildInfoImpl.h                           (C) 2000-2026 */
/*                                                                           */
/* Informations pour construire une instance d'une application.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_ARCCOREAPPLICATIONBUILDINFOIMPL_H
#define ARCCORE_COMMON_INTERNAL_ARCCOREAPPLICATIONBUILDINFOIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/ArccoreApplicationBuildInfo.h"
#include "arccore/common/internal/FieldProperty.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_COMMON_EXPORT ApplicationCoreBuildInfo::CoreImpl
{
 public:

  template <typename T> using FieldProperty = PropertyImpl::FieldProperty<T>;

  CoreImpl();

 public:

  String getValue(const UniqueArray<String>& env_values, const String& param_name,
                  const String& default_value);
  void addKeyValue(const String& name, const String& value);

 public:

  FieldProperty<StringList> m_task_implementation_services;
  FieldProperty<StringList> m_thread_implementation_services;
  FieldProperty<Int32> m_nb_task_thread;

 private:

  PropertyImpl::PropertyKeyValues m_property_key_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

