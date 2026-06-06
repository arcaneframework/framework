// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceBuildInfo.h                                          (C) 2000-2025 */
/*                                                                           */
/* Structure containing the information to create a service.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SERVICEBUILDINFO_H
#define ARCANE_CORE_SERVICEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/VersionInfo.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ServiceProperty.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information for creating a service.
 *
 * Instances of this class are internal to Arcane. Generally, you must
 * use the ServiceBuildInfo class.
 *
 * The different fields of this class are not all valid. Their validity
 * depends on the constructor used. Only application() is always valid. It
 * is possible to know the valid fields via the value of creationType().
 */
class ARCANE_CORE_EXPORT ServiceBuildInfoBase
{
 public:

  /*!@{
   * \name Constructors.
   *
   * The different constructors allow defining the service type
   * (eServiceType).
   */

  /*!
   * \brief Service associated with an application \a IApplication.
   *
   * The service will be of type #ST_Application.
   */
  explicit ServiceBuildInfoBase(IApplication* app);

  /*!
   * \brief Service associated with a session \a ISession.
   *
   * The service will be of type #ST_Session.
   */
  explicit ServiceBuildInfoBase(ISession* session);

  /*!
   * \brief Service associated with a subdomain \a ISubDomain.
   *
   * The service will be of type #ST_SubDomain.
   * Also provides access to the mesh property (\a mesh())
   */
  explicit ServiceBuildInfoBase(ISubDomain* sd);

  /*!
   * \brief Service associated with a mesh \a mesh.
   *
   * The service will be of type #ST_SubDomain.
   */
  ServiceBuildInfoBase(ISubDomain* sd, IMesh* mesh);

  /*!
   * \brief Service associated with a mesh \a mesh_handle.
   *
   * The service will be of type #ST_SubDomain.
   */
  explicit ServiceBuildInfoBase(const MeshHandle& mesh_handle);

  /*!
   * \brief Service associated with a mesh \a mesh_handle.
   *
   * The service will be of type #ST_SubDomain.
   */
  ServiceBuildInfoBase(ISubDomain* sd, const MeshHandle& mesh_handle);

  /*!
   * \brief Service associated with a dataset option \a co.
   *
   * The service will be of type #ST_CaseOption.
   * Also provides access to the mesh (mesh()) and
   * subdomain (subDomain()) properties.
   */
  explicit ServiceBuildInfoBase(ICaseOptions* co);

  /*!
   * \brief Service associated with a dataset option \a co.
   *
   * The service will be of type #ST_CaseOption.
   * Also provides access to the mesh (mesh()) and
   * subdomain (subDomain()) properties.
   */
  ServiceBuildInfoBase(ISubDomain* sd, ICaseOptions* co);

  /*!
   * \brief Service associated with a mesh \a mesh.
   *
   * The service will be of type #ST_SubDomain.
   *
   * \deprecated Use ServiceBuildInfoBase(const MeshHandle&) instead.
   */
  ARCCORE_DEPRECATED_2020("Use ServiceBuildInfoBase(const MeshHandle&) instead")
  explicit ServiceBuildInfoBase(IMesh* mesh);

  //!@}

 public:

  /*!
   * \brief Access to the associated \a IApplication.
   *
   * The instance is never null regardless of the service.
   */
  IApplication* application() const { return m_application; }

  /*!
   * \brief Access to the associated \a ISession.
   *
   * \pre creationType() & (#ST_CaseOption|#ST_SubDomain|#ST_Session)
   */
  ISession* session() const { return m_session; }

  /*!
   * \brief Access to the associated \a ISubDomain.
   *
   * \pre creationType() & (#ST_CaseOption|#ST_SubDomain)
   */
  ISubDomain* subDomain() const { return m_sub_domain; }

  /*!
   * \brief Access to the associated \a IMesh.
   *
   * \pre creationType() & (#ST_CaseOption|#ST_SubDomain)
   */
  IMesh* mesh() const;

  /*!
   * \brief Access to the associated mesh handle \a MeshHandle.
   *
   * \pre creationType() & (#ST_CaseOption|#ST_SubDomain)
   */
  const MeshHandle& meshHandle() const { return m_mesh_handle; }

  /*!
   * \brief Access to the associated \a ICaseOptions.
   *
   * \pre creationType() & #ST_CaseOption
   */
  ICaseOptions* caseOptions() const { return m_case_options; }

  /*!
   * \brief Access to the parent instance that created this instance.
   */
  IBase* serviceParent() const { return m_service_parent; }

  //! Type of service that can be created by this instance
  eServiceType creationType() const { return m_creation_type; }

 protected:
 private:

  IApplication* m_application = nullptr;
  ISession* m_session = nullptr;
  ISubDomain* m_sub_domain = nullptr;
  MeshHandle m_mesh_handle;
  ICaseOptions* m_case_options = nullptr;
  IBase* m_service_parent = nullptr;
  eServiceType m_creation_type = ST_None;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Service
 * \brief Structure containing the information to create a service.
 */
class ARCANE_CORE_EXPORT ServiceBuildInfo
: public ServiceBuildInfoBase
{
 public:

  ServiceBuildInfo(IServiceInfo* service_info, const ServiceBuildInfoBase& base);

 public:

  //! Access to the associated \a IServiceInfo
  IServiceInfo* serviceInfo() const { return m_service_info; }

 private:

  IServiceInfo* m_service_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
