// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCodeService.cc                                        (C) 2000-2019 */
/*                                                                           */
/* Service de code générique Arcane.                                         */
/* Ce service est une recopie locale de celui de tests pour le driver        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/CodeService.h"
#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/ISession.h"
#include "arcane/ISubDomain.h"
#include "arcane/IParallelMng.h"
#include "arcane/Service.h"

#include "arcane/impl/TimeLoopReader.h"
#include "arcane/impl/ArcaneSession.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneCodeService
: public CodeService
{
 public:

  ArcaneCodeService(const ServiceBuildInfo& sbi);
  virtual ~ArcaneCodeService();

 public:

  virtual bool parseArgs(StringList& args);
  virtual ISession* createSession();
  virtual void initCase(ISubDomain* sub_domain,bool is_continue);

 public:

  void build() {}

 protected:

  virtual void _preInitializeSubDomain(ISubDomain* sd);

 public:

  static Internal::ServiceInfo service_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Internal::ServiceInfo
ArcaneCodeService::service_info("ArcaneCodeService",VersionInfo(1,0,1),
                                IServiceInfo::Dim1|IServiceInfo::Dim2|IServiceInfo::Dim3);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT Ref<ICodeService>
createArcaneCodeService(IApplication* app)
{
  ServiceBuildInfoBase s(app);
  auto x = new ArcaneCodeService(ServiceBuildInfo(&ArcaneCodeService::service_info,s));
  x->build();
  return makeRef<ICodeService>(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCodeService::
ArcaneCodeService(const ServiceBuildInfo& sbi)
: CodeService(sbi)
{
  _addExtension(String("arc"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneCodeService::
~ArcaneCodeService()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCodeService::
_preInitializeSubDomain(ISubDomain*)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneCodeService::
initCase(ISubDomain* sub_domain,bool is_continue)
{
  {
    TimeLoopReader stl(_application());
    stl.readTimeLoops();
    stl.registerTimeLoops(sub_domain);
    stl.setUsedTimeLoop(sub_domain);
  }
  CodeService::initCase(sub_domain,is_continue);
  if (sub_domain->parallelMng()->isMasterIO())
    sub_domain->session()->writeExecInfoFile();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISession* ArcaneCodeService::
createSession()
{
  ArcaneSession* session = new ArcaneSession(_application());
  session->build();
  _application()->addSession(session);
  return session;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneCodeService::
parseArgs(StringList& args)
{
  ARCANE_UNUSED(args);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
