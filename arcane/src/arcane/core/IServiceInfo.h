// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IServiceInfo.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface of service information.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERVICEINFO_H
#define ARCANE_CORE_ISERVICEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Service
 * \brief Interface for service or module information.
 */
class ARCANE_CORE_EXPORT IServiceInfo
{
 public:

  static const Integer Dim1 = 1;
  static const Integer Dim2 = 2;
  static const Integer Dim3 = 4;

 public:

  virtual ~IServiceInfo() = default; //!< Frees resources

 public:

  //! Local part of the service name
  virtual String localName() const = 0;

  //! Service name namespace
  virtual String namespaceURI() const = 0;

  //! Service version
  virtual VersionInfo version() const = 0;

  //! Version of the axl file describing this service
  virtual Real axlVersion() const = 0;

  //! Indicates if the service is usable in dimension \a n.
  virtual bool allowDimension(Integer n) const = 0;

  /*! \brief Adds the name interface \a name to the interfaces
   * implemented by this service.
   */
  virtual void addImplementedInterface(const String& name) = 0;

  //! List of names of classes implemented by this service
  virtual StringCollection implementedInterfaces() const = 0;

  //! Name of the file containing the dataset (null if none)
  virtual const String& caseOptionsFileName() const = 0;

  //! List of service factories
  virtual ServiceFactory2Collection factories() const = 0;

  //! Factory for singleton services (nullptr if not supported)
  virtual Internal::ISingletonServiceFactory* singletonFactory() const = 0;

  /*! \brief Name of the service XML element for the language \a lang.
   * If \a lang is null, returns the default name.
   */
  virtual String tagName(const String& lang) const = 0;

  //! Information on factories available for this service
  virtual IServiceFactoryInfo* factoryInfo() const = 0;

  /*!
   * \brief Indicates where the service can be used.
   *
   * It is a combination of eServiceType values.
   */
  virtual int usageType() const = 0;

  //! Content of the AXL file associated with this service or module
  virtual const FileContent& axlContent() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
