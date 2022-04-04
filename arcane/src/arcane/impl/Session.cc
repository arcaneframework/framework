// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Session.cc                                                  (C) 2000-2014 */
/*                                                                           */
/* Session.                                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/CriticalSection.h"

#include "arcane/IApplication.h"
#include "arcane/IIOMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/ICaseDocument.h"
#include "arcane/XmlNode.h"
#include "arcane/CaseNodeNames.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMainFactory.h"
#include "arcane/IParallelSuperMng.h"
#include "arcane/IServiceMng.h"
#include "arcane/SubDomainBuildInfo.h"

#include "arcane/impl/Session.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ISubDomain*
arcaneCreateSubDomain(ISession* session,const SubDomainBuildInfo& sdbi);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ISession*
arcaneCreateSession(IApplication* sm)
{
  Session* s = new Session(sm);
  s->build();
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Session.
 */
class Session::Impl
{
 public:

  Impl(IApplication* app)
  : m_application(app), m_namespace_uri(arcaneNamespaceURI())
    {}
  ~Impl() {}

 public:

  IApplication* m_application; //!< Superviseur
  String m_filename; //!< Fichier de configuration
  SubDomainList m_sub_domains;
  ScopedPtrT<IServiceMng> m_service_mng; //!< Gestionnaire des services
  String m_namespace_uri;
  String m_local_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Session::
Session(IApplication* app)
: TraceAccessor(app->traceMng())
, m_p(0)
{
  m_p = new Impl(app);
  m_p->m_local_name = "Session";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Session::
~Session()
{
  for( SubDomainList::Enumerator i(m_p->m_sub_domains); ++i; ){
    (*i)->destroy();
  }
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Session::
build()
{
  IMainFactory* mf = m_p->m_application->mainFactory();
  m_p->m_service_mng = mf->createServiceMng(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * La classe dérivée peut réimplémenter cette méthode pour spécifier
 * son propre mécanisme de versionnage des jeux de données. Sans
 * implémentation particulière, cette méthode retourne toujours \a false
 *
 * \retval true si la version est correcte
 * \retval false sinon
 */
bool Session::
checkIsValidCaseVersion(const String& version)
{
  ARCANE_UNUSED(version);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* Session::
createSubDomain(const SubDomainBuildInfo& sdbi)
{
  IParallelSuperMng* sm = application()->parallelSuperMng();
  ISubDomain* s = 0;
  {
    CriticalSection cs(sm->threadMng());
    s = arcaneCreateSubDomain(this,sdbi);
    //TODO: Utiliser le rang local pour ranger dans l'ordre
    m_p->m_sub_domains.add(s);
  }
  s->initialize();
  // Initialisation spécifique à la classe dérivée si besoin
  _initSubDomain(s);
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Session::
doAbort()
{
  IParallelSuperMng* psm = m_p->m_application->parallelSuperMng();
  if (psm)
    psm->tryAbort();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IBase* Session::
objectParent() const
{
  return m_p->m_application;
}

String Session::objectNamespaceURI() const { return m_p->m_namespace_uri; }
String Session::objectLocalName() const { return m_p->m_local_name; }
VersionInfo Session::objectVersion() const { return VersionInfo(1,0,0); }
IServiceMng* Session::serviceMng() const { return m_p->m_service_mng.get(); }
IRessourceMng* Session::ressourceMng() const { return 0; }
IApplication* Session::application() const { return m_p->m_application; }
ITraceMng* Session::traceMng() const { return TraceAccessor::traceMng(); }
const String& Session::fileName() const { return m_p->m_filename; }
SubDomainCollection Session::subDomains() { return m_p->m_sub_domains; }
IApplication* Session::_application() const { return m_p->m_application; }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
