// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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

#include "arcane/core/IApplication.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/CaseNodeNames.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IParallelSuperMng.h"
#include "arcane/core/IServiceMng.h"
#include "arcane/core/SubDomainBuildInfo.h"

#include "arcane/impl/Session.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ISubDomain*
arcaneCreateSubDomain(ISession* session, const SubDomainBuildInfo& sdbi);

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
  : m_application(app)
  , m_namespace_uri(arcaneNamespaceURI())
  {}
  ~Impl() {}

 public:

  IApplication* m_application; //!< Supervisor
  String m_filename; //!< Configuration file
  SubDomainList m_sub_domains;
  ScopedPtrT<IServiceMng> m_service_mng; //!< Service manager
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
  for (SubDomainList::Enumerator i(m_p->m_sub_domains); ++i;) {
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
 * The derived class can re-implement this method to specify
 * its own dataset versioning mechanism. Without specific implementation, this method always returns \a false
 *
 * \retval true if the version is correct
 * \retval false otherwise
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
    s = arcaneCreateSubDomain(this, sdbi);
    //TODO: Use the local rank to sort in order
    m_p->m_sub_domains.add(s);
  }
  s->initialize();
  // Specific initialization for the derived class if needed
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

String Session::objectNamespaceURI() const
{
  return m_p->m_namespace_uri;
}
String Session::objectLocalName() const
{
  return m_p->m_local_name;
}
VersionInfo Session::objectVersion() const
{
  return VersionInfo(1, 0, 0);
}
IServiceMng* Session::serviceMng() const
{
  return m_p->m_service_mng.get();
}
IRessourceMng* Session::ressourceMng() const
{
  return 0;
}
IApplication* Session::application() const
{
  return m_p->m_application;
}
ITraceMng* Session::traceMng() const
{
  return TraceAccessor::traceMng();
}
const String& Session::fileName() const
{
  return m_p->m_filename;
}
SubDomainCollection Session::subDomains()
{
  return m_p->m_sub_domains;
}
IApplication* Session::_application() const
{
  return m_p->m_application;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
