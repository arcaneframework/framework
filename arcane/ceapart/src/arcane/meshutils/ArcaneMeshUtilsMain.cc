// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Exception.h"

#include "arcane/FactoryService.h"
#include "arcane/CodeService.h"
#include "arcane/impl/ArcaneMain.h"
#include "arcane/impl/Session.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de code pour les utilitaires de maillage.
 */
class ArcaneMeshUtilsCodeService
: public Arcane::CodeService
{
 public:

  ArcaneMeshUtilsCodeService(const ServiceBuildInfo& sbi)
  : CodeService(sbi)
  {
    _addExtension("msh");
  }

 public:

  void build() {}
  ISession* createSession() override
  {
    ISession* session = new Session(_application());
    _application()->addSession(session);
    return session;
  }
  bool parseArgs(StringList& args) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneMeshUtilsCodeService::
parseArgs(StringList& args)
{
  ARCANE_UNUSED(args);
  ITraceMng* trace = _application()->traceMng();
  trace->info() << "** PARSE ARGS";
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(ArcaneMeshUtilsCodeService,
                                    ICodeService,
                                    ArcaneMeshUtilsCode);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int
_mainHelper(int argc,char* argv[])
{
  int r = 0;
  ArcaneMain::arcaneInitialize();
  {
    ApplicationInfo app_info(&argc,&argv,"ArcaneMeshUtils",VersionInfo(1,0,0));
    r = ArcaneMain::arcaneMain(app_info);
  }
  ArcaneMain::arcaneFinalize();
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int
main(int argc,char* argv[])
{
  int r = 0;
  int r2 = arcaneCallFunctionAndCatchException([&](){ r = _mainHelper(argc,argv); });
  if (r2!=0)
    return r2;
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
