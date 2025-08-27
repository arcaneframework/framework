// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceBuildInfo.h                                          (C) 2000-2022 */
/*                                                                           */
/* Structure contenant les informations pour créer un service.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SERVICEBUILDINFO_H
#define ARCANE_SERVICEBUILDINFO_H
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
 * \brief Informations pour créer un service.
 *
 * Les instances de cette classe sont internes à Arcane. En général il faut
 * utiliser la classe ServiceBuildInfo.
 *
 * Les différents champs de cette classe ne sont pas tous valides. Leur validité
 * dépend du constructeur utilisé. Seul application() est toujours valide.
 * Il est possible de connaitre les champs valides via la valeur de
 * creationType().
 */
class ARCANE_CORE_EXPORT ServiceBuildInfoBase
{
 public:

  /*!@{
   * \name Constructeurs.
   *
   * Les différents constructeurs permettent de définir le type de service
   * (eServiceType).
   */

  /*!
   * \brief Service associé à une application \a IApplication.
   *
   * Le service sera de type #ST_Application.
   */
  explicit ServiceBuildInfoBase(IApplication* app);

  /*!
   * \brief Service associé à une session \a ISession.
   *
   * Le service sera de type #ST_Session.
   */
  explicit ServiceBuildInfoBase(ISession* session);

  /*!
   * \brief Service associé à un sous-domaine \a ISubDomain.
   *
   * Le service sera de type #ST_SubDomain.
   * Donne aussi accès à la propriété de  maillage (\a mesh())
   */
  explicit ServiceBuildInfoBase(ISubDomain* sd);

  /*!
   * \brief Service associé à un maillage \a mesh.
   *
   * Le service sera de type #ST_SubDomain.
   */
  ServiceBuildInfoBase(ISubDomain* sd, IMesh* mesh);

  /*!
   * \brief Service associé à un maillage \a mesh_handle.
   *
   * Le service sera de type #ST_SubDomain.
   */
  explicit ServiceBuildInfoBase(const MeshHandle& mesh_handle);

  /*!
   * \brief Service associé à un maillage \a mesh_handle.
   *
   * Le service sera de type #ST_SubDomain.
   */
  ServiceBuildInfoBase(ISubDomain* sd, const MeshHandle& mesh_handle);

  /*!
   * \brief Service associé à une option du jeu de données \a co.
   *
   * Le service sera de type #ST_CaseOption.
   * Donne aussi accès aux propriétés de maillage (mesh()) et
   * sous-domaine (subDomain()).
   */
  explicit ServiceBuildInfoBase(ICaseOptions* co);

  /*!
   * \brief Service associé à une option du jeu de données \a co.
   *
   * Le service sera de type #ST_CaseOption.
   * Donne aussi accès aux propriétés de maillage (mesh()) et
   * sous-domaine (subDomain()).
   */
  ServiceBuildInfoBase(ISubDomain* sd, ICaseOptions* co);

  /*!
   * \brief Service associé à un maillage \a mesh.
   *
   * Le service sera de type #ST_SubDomain.
   *
   * \deprecated Utiliser ServiceBuildInfoBase(const MeshHandle&) à la place.
   */
  ARCCORE_DEPRECATED_2020("Use ServiceBuildInfoBase(const MeshHandle&) instead")
  explicit ServiceBuildInfoBase(IMesh* mesh);

  //!@}

 public:

  /*!
   * \brief Accès à l'application \a IApplication associé.
   *
   * L'instance n'est jamais nulle quel que soit le service.
   */
  IApplication* application() const { return m_application; }

  /*!
   * \brief Accès au \a ISession associé.
   *
   * \pre creationType() & (#ST_CaseOption|#ST_SubDomain|#ST_Session)
   */
  ISession* session() const { return m_session; }

  /*!
   * \brief Accès au \a ISubDomain associé.
   *
   * \pre creationType() & (#ST_CaseOption|#ST_SubDomain)
   */
  ISubDomain* subDomain() const { return m_sub_domain; }

  /*!
   * \brief Accès au \a IMesh associé.
   *
   * \pre creationType() & (#ST_CaseOption|#ST_SubDomain)
   */
  IMesh* mesh() const;

  /*!
   * \brief Accès au handle de maillage \a MeshHandle associé.
   *
   * \pre creationType() & (#ST_CaseOption|#ST_SubDomain)
   */
  const MeshHandle& meshHandle() const { return m_mesh_handle; }

  /*!
   * \brief Accès au \a ICaseOptions associé.
   *
   * \pre creationType() & #ST_CaseOption
   */
  ICaseOptions* caseOptions() const { return m_case_options; }

  /*!
   * \brief Accès à l'instance parente qui a créée cette instance.
   */
  IBase* serviceParent() const { return m_service_parent; }

  //! Type du service pouvant être créé par cette instance
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

