// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceBuildInfo.h                                          (C) 2000-2020 */
/*                                                                           */
/* Structure contenant les informations pour créer un service.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SERVICEBUILDINFO_H
#define ARCANE_SERVICEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/VersionInfo.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/ServiceProperty.h"
#include "arcane/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IServiceInfo;
class IApplication;
class ISession;
class ISubDomain;
class ICaseOptions;
class IMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Structure contenant les informations pour créer un service.
 *
 * Les différents champs ne sont pas tous valides. Leur validité dépend
 * du constructeur utilisé. Seul application() est toujours valide.
 */
class ARCANE_CORE_EXPORT ServiceBuildInfoBase
{
 public:
  /*!@{ @name Constructeur d'information de construction de service
   *
   * Les différents constructeurs ont pour objectif la construction de
   * services dans différents contextes. Il ne sont globalement par
   * interchangeables car ne donneront pas accès à toutes les
   * propriétés possibles.
   *
   * Les \a ServiceBuildInfo sont principalement utilisés 
   * dans les *ServiceFactory
   */
  //! Constructeur à partir d'un \a IServiceInfo et d'un \a IApplication
  explicit ServiceBuildInfoBase(IApplication* app);
  //! Constructeur à partir d'un \a IServiceInfo et d'une \a ISession
  explicit ServiceBuildInfoBase(ISession* session);
  //! Constructeur à partir d'un \a IServiceInfo et d'un \a ISubDomain
  /*! Donne aussi accès à la propriété de  maillage (\a mesh()) */
  explicit ServiceBuildInfoBase(ISubDomain* sd);
  //! Constructeur à partir d'un \a IServiceInfo et d'un \a IMesh
  ARCCORE_DEPRECATED_2020("Use ServiceBuildInfoBase(ISubDomain*,const MeshHandle&) instead")
  explicit ServiceBuildInfoBase(IMesh* mesh);
  //! Constructeur à partir d'un \a IServiceInfo et d'un \a IMesh
  ServiceBuildInfoBase(ISubDomain* sd,IMesh* mesh);
  //! Constructeur à partir d'un \a IServiceInfo et d'un maillage
  ARCCORE_DEPRECATED_2020("Use ServiceBuildInfoBase(ISubDomain*,const MeshHandle&) instead")
  explicit ServiceBuildInfoBase(const MeshHandle& mesh_handle);
  //! Constructeur à partir d'un \a ISubDomain et d'un maillage
  explicit ServiceBuildInfoBase(ISubDomain* sd,const MeshHandle& mesh_handle);
  //! Constructeur à partir d'un \a IServiceInfo et d'un \a ICaseOptions
  /*! Donne aussi accès aux propriétés de maillage (\a mesh()) et
   * sous-domaine (\a subDomain()) */
  ARCCORE_DEPRECATED_2020("Use ServiceBuildInfoBase(ISubDomain*,ICaseOptions*) instead")
  explicit ServiceBuildInfoBase(ICaseOptions* co);
  ServiceBuildInfoBase(ISubDomain* sd,ICaseOptions* co);
  //!@}

 public:

  //! Accès au \a ISubDomain associé
  ISubDomain* subDomain() const { return m_sub_domain; }
  //! Accès au \a IMesh associé
  IMesh* mesh() const;
  //! Accès au \a IMesh associé
  const MeshHandle& meshHandle() const { return m_mesh_handle; }
  //! Accès au \a IApplication associé
  IApplication* application() const { return m_application; }
  //! Accès au \a ISession associé
  ISession* session() const { return m_session; }
  //! Accès au \a ICaseOptions associé
  ICaseOptions* caseOptions() const { return m_case_options; }
  //! Accès au \a IBase associé
  /*! Méthode disponible pour tout contexte de création */
  IBase* serviceParent() const { return m_service_parent; }

  //! Type du service pouvant être créé par cette instance
  eServiceType creationType() const { return m_creation_type; }

 protected:
  IApplication* m_application = nullptr;
  ISession* m_session = nullptr;
  ISubDomain* m_sub_domain = nullptr;
  MeshHandle m_mesh_handle;
  ICaseOptions* m_case_options = nullptr;
  IBase* m_service_parent = nullptr;
  eServiceType m_creation_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Service
 * \brief Structure contenant les informations pour créer un service.
 */
class ARCANE_CORE_EXPORT ServiceBuildInfo
: public ServiceBuildInfoBase
{
 public:
  ServiceBuildInfo(IServiceInfo* service_info, const ServiceBuildInfoBase& base);

 public:
  //! Accès au \a IServiceInfo associé
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

