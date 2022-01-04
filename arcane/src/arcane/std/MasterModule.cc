// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MasterModule.cc                                             (C) 2000-2011 */
/*                                                                           */
/* Module maître.                                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/ITimeLoopService.h"
#include "arcane/IModuleMaster.h"

#include "arcane/std/Master_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module maître
 */
class MasterModule
: public ArcaneMasterObject
{
 public:

  MasterModule(const ModuleBuildInfo& cb);
  ~MasterModule();

 public:
	
  virtual VersionInfo versionInfo() const { return VersionInfo(1,1,0); }

 public:

  virtual void masterBuild();

 private:

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_MASTER(MasterModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MasterModule::
MasterModule(const ModuleBuildInfo& mb)
: ArcaneMasterObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MasterModule::
~MasterModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MasterModule::
masterBuild()
{
  IModuleMaster* master = subDomain()->moduleMaster();
  for( Integer i=0, is=options()->globalService.size(); i<is; ++i ){
    ITimeLoopService* service = options()->globalService[i];
    master->addTimeLoopService(service);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
