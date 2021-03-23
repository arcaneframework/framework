// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <mono/jit/jit.h>
#include <mono/metadata/environment.h>
#include <mono/metadata/mono-config.h>
#include <mono/metadata/threads.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/Concurrency.h"
#include "arcane/ObserverPool.h"

namespace
{
bool dotnet_verbose = false;
bool global_is_running = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Point d'entrée de l'exécutable pour 'mono'.

extern "C" int mono_main(int argc,char* argv[]);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneMonoThreadAttachCallback
{
 public:
  void init()
  {
    m_observers.addObserver(this,&ArcaneMonoThreadAttachCallback::_callback,
                            Arcane::TaskFactory::createThreadObservable());
  }
 private:
  void _callback()
  {
#ifndef ARCANE_MONO_NO_THREAD_ATTACH
    if (dotnet_verbose)
      std::cout << "MONO_THREAD_ATTACH CALLBACK !\n";
    mono_thread_attach (mono_get_root_domain ());
#endif
  }
 private:
  Arcane::ObserverPool m_observers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int
_arcane_mono_main_internal(int argc, char* argv[],const char* assembly_name)
{
  if (global_is_running)
    ARCANE_FATAL("mono wrapper has already been launched");
  global_is_running = true;
  int new_argc = argc+1;
  char** new_argv = new char*[new_argc+1];
  new_argv[0] = strdup(argv[0]);
  for( int i=1; i<argc; ++i )
    new_argv[i+1] = argv[i];
  new_argv[1] = strdup(assembly_name);
  Arcane::String verbose_str = Arcane::platform::getEnvironmentVariable("ARCANE_DEBUG_DOTNET");
  dotnet_verbose = !verbose_str.null();

  if (dotnet_verbose){
    std::cout << "ArcaneMono: ASSEMBLY NAME is '" << assembly_name << "' argc=" << argc << "\n";
    for( int i=0; i<new_argc; ++i )
      std::cout<< "ArcaneMono: Arg i=" << i << " V=" << new_argv[i] << '\n';
    std::cout.flush();
  }

  ArcaneMonoThreadAttachCallback mtac;
  mtac.init();

  int retval = mono_main(new_argc,new_argv);
  global_is_running = false;
  return retval;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Obsolète: utiliser 'arcane_mono_main2' à la place.
extern "C" ARCANE_EXPORT int
arcane_mono_main(int argc, char* argv[],const char* assembly_name)
{
  if (!assembly_name || assembly_name[0]=='\0')
    assembly_name = "Arcane.Main.dll";
  return _arcane_mono_main_internal(argc,argv,assembly_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" ARCANE_EXPORT int
arcane_mono_main2(const Arcane::CommandLineArguments& cmd_args,
                  const Arcane::String& orig_assembly_name)
{
  Arcane::String assembly_name = orig_assembly_name;
  if (assembly_name.empty())
    assembly_name = "Arcane.Main.dll";

  int argc = *(cmd_args.commandLineArgc());
  char** argv = *(cmd_args.commandLineArgv());
  const char* assembly_name_str = assembly_name.localstr();
  return _arcane_mono_main_internal(argc,argv,assembly_name_str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
arcane_mono_launch()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
