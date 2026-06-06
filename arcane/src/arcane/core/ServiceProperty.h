// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceProperty.h                                           (C) 2000-2022 */
/*                                                                           */
/* Properties of a service.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SERVICEPROPERTY_H
#define ARCANE_SERVICEPROPERTY_H
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
 * \brief Properties for service factories.
 *
 * These are flags used with the binary OR operator (|).
 */
enum eServiceFactoryProperties
{
  //! No specific property
  SFP_None = 0,
  //! Indicates that the service is a singleton
  SFP_Singleton = 1,
  //! Indicates that the service loads automatically.
  SFP_Autoload = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Service type.
 *
 * This enumeration allows knowing where a service can be created.
 *
 * These are flags used with the binary OR operator (|).
 * A service can therefore be available in several places. For example,
 * it can be present as a dataset option (#ST_CaseOption) and also
 * at the subdomain level (#ST_SubDomain). In this latter case,
 * it can be created via the ServiceBuilder class.
 *
 * \note This type must correspond to the corresponding C# type
 */
enum eServiceType
{
  ST_None = 0,
  //! The service is used at the application level
  ST_Application = 1,
  //! The service is used at the session level
  ST_Session = 2,
  //! The service is used at the subdomain level
  ST_SubDomain = 4,
  //! The service is used at the dataset level.
  ST_CaseOption = 8,
  // NOTE: This value is not yet used.
  //! The service is used with an explicitly specified mesh.
  ST_Mesh = 16
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Service creation properties.
 *
 * This class is used in service registration macros
 * and can therefore be instantiated as a global variable before entering
 * the code's main(). It should therefore only contain Plain Object Data (POD) fields.
 *
 * Generally, instances of this class are used when
 * registering a service via the ARCANE_REGISTER_SERVICE() macro.
 *
 * In the constructor, the \a type and \a properties parameters
 * can use a combination of enumerated values. For example,
 * to specify a service that can be used both in the
 * dataset and at the subdomain level, you can do the following:
 *
 * \code
 * ServiceProperty("ServiceName",ST_SubDomain|ST_CaseOption);
 * \endcode
 */
class ARCANE_CORE_EXPORT ServiceProperty
{
 public:

  /*!
   * \brief Constructs an instance for a service named \a aname and of type \a atype
   * with properties \a properties.
   */
  ServiceProperty(const char* aname, int atype, eServiceFactoryProperties aproperties) ARCANE_NOEXCEPT
  : m_name(aname)
  , m_type(atype)
  , m_properties(aproperties)
  {
  }

  //! Constructs an instance for a service named \a aname and of type \a atype
  ServiceProperty(const char* aname, int atype) ARCANE_NOEXCEPT
  : m_name(aname)
  , m_type(atype)
  , m_properties(SFP_None)
  {
  }

  //! Constructs an instance for a service named \a aname and of type \a atype
  ServiceProperty(const char* aname, eServiceType atype) ARCANE_NOEXCEPT
  : m_name(aname)
  , m_type((int)atype)
  , m_properties(SFP_None)
  {
  }

 public:

  //! Service name.
  const char* name() const { return m_name; }

  //! Service type (combination of eServiceType)
  int type() const { return m_type; }

  //! Service properties (combination of eServiceFactoryProperties)
  eServiceFactoryProperties properties() const { return m_properties; }

 private:

  const char* m_name;
  int m_type;
  eServiceFactoryProperties m_properties;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
