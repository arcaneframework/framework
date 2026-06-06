// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Configuration.h                                             (C) 2000-2020 */
/*                                                                           */
/* Management of execution configuration options.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CONFIGURATION_H
#define ARCANE_CONFIGURATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for a configuration section.
 *
 * This interface allows retrieving the values of a configuration option.
 */
class ARCANE_CORE_EXPORT IConfigurationSection
{
 public:

  virtual ~IConfigurationSection() {} //!< Frees resources

 public:

  virtual Int32 value(const String& name,Int32 default_value) const =0;
  virtual Int64 value(const String& name,Int64 default_value) const =0;
  virtual Real value(const String& name,Real default_value) const =0;
  virtual bool value(const String& name,bool default_value) const =0;
  virtual String value(const String& name,const String& default_value) const =0;
  virtual String value(const String& name,const char* default_value) const =0;

  virtual Integer valueAsInteger(const String& name,Integer default_value) const =0;
  virtual Int32 valueAsInt32(const String& name,Int32 default_value) const =0;
  virtual Int64 valueAsInt64(const String& name,Int64 default_value) const =0;
  virtual Real valueAsReal(const String& name,Real default_value) const =0;
  virtual bool valueAsBool(const String& name,bool default_value) const =0;
  virtual String valueAsString(const String& name,const String& default_value) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface for a configuration.
 */
class ARCANE_CORE_EXPORT IConfiguration
{
 public:

  virtual ~IConfiguration() {} //!< Frees resources

 public:

  /*!
   * \brief Creates a configuration section.
   *
   * The returned instance must be destroyed by the delete operator.
   */
  virtual IConfigurationSection* createSection(const String& name) const =0;

  /*!
   * \brief Main section.
   *
   * The returned instance remains the property of this instance
   * and should not be destroyed.
   */
  virtual IConfigurationSection* mainSection() const =0;

  /*!
   * \brief Adds a value to the configuration.
   *
   * Adds the value \a value for the name \a name to the configuration.
   * The new value will have priority \a priority. If a value for name
   * \a name already exists, it is replaced by \a value if \a priority
   * is lower than the current priority.
   */
  virtual void addValue(const String& name,const String& value,Integer priority) =0;
  
  /*!
   * \brief Clones this configuration.
   */
  virtual IConfiguration* clone() const =0;

  /*!
   * \brief Merges this configuration with configuration \a c.
   *
   * If an option exists in both this configuration and \a c, the one
   * with the lowest priority is kept.
   */
  virtual void merge(const IConfiguration* c) =0;

  //! Displays the values of the configuration parameters via traceMng()
  virtual void dump() const =0;

  //! Displays the values of the configuration parameters to the stream o
  virtual void dump(std::ostream& ostr) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Configuration manager.
 */
class ARCANE_CORE_EXPORT IConfigurationMng
{
 public:

  virtual ~IConfigurationMng() {} //!< Frees resources

 public:

  //! Default configuration.
  virtual IConfiguration* defaultConfiguration() const =0;

  /*!
   * \brief Creates a new configuration.
   *
   * The returned instance must be destroyed by the delete operator.
   */
  virtual IConfiguration* createConfiguration() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
