// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneCodeService.cc                                        (C) 2000-2012 */
/*                                                                           */
/* Arcane generic code service.                                              */
/* This service is a local copy of the test service for the driver.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CodeService.h"
#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/core/ISession.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/Service.h"

#include "arcane/impl/TimeLoopReader.h"

#include "arcane/std/ArcaneSession.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
  virtual void initCase(ISubDomain* sub_domain, bool is_continue);

 public:

  void build() {}

 protected:

  virtual void _preInitializeSubDomain(ISubDomain* sd);

 private:
};

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
initCase(ISubDomain* sub_domain, bool is_continue)
{
  {
    TimeLoopReader stl(_application());
    stl.readTimeLoops();
    stl.registerTimeLoops(sub_domain);
    stl.setUsedTimeLoop(sub_domain);
  }
  CodeService::initCase(sub_domain, is_continue);
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

ARCANE_REGISTER_APPLICATION_FACTORY(ArcaneCodeService, ICodeService, ArcaneCode);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
