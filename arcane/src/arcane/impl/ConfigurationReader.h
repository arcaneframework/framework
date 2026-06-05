// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConfigurationReader.h                                       (C) 2000-2020 */
/*                                                                           */
/* Configuration file readers.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_CONFIGURATIONREADER_H
#define ARCANE_IMPL_CONFIGURATIONREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XmlNode;
class IConfiguration;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Configuration file readers.
 */
class ARCANE_IMPL_EXPORT ConfigurationReader
: public TraceAccessor
{
 public:

  enum Priority
  {
    P_CaseDocument = 10,
    P_TimeLoop = 50,
    P_Global = 100,
    P_GlobalRuntime = 110
  };

 public:

  ConfigurationReader(ITraceMng* tm, IConfiguration* config)
  : TraceAccessor(tm)
  , m_configuration(config)
  {}

  /*!
   * \brief Adds values to the configuration.
   *
   * Adds to the configuration the values contained in the child elements
   * of \a element. The elements considered are those of the following form:
   * - <add name="ConfigName" value="ConfigValue" />
   *
   * The IConfiguration::addValue() method is called for each value with
   * the priority \a priority.
   */
  void addValuesFromXmlNode(const XmlNode& element, Integer priority);

  /*!
   * \brief Adds values to the configuration.
   *
   * Adds to the configuration the values contained in the child fields
   * of \a jv.
   *
   * Array elements are not considered. The call is recursive and the
   * children of \a jv are considered. This is equivalent to creating a
   * subsection in the configuration.
   *
   * The IConfiguration::addValue() method is called for each value with
   * the priority \a priority.
   */
  void addValuesFromJSON(const JSONValue& jv, Integer priority);

 private:

  IConfiguration* m_configuration;

 private:

  void _addValuesFromJSON(const JSONValue& jv, Integer priority, const String& base_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
