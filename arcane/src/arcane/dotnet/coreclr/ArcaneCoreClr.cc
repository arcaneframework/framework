// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Exception.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/core/Directory.h"

// Standard headers
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>

// NOTE: à partir de '.Net 6', ces fichiers sont inclus dans le SDK
// au même endroit que 'libnethost.so'

// Provided by the AppHost NuGet package and installed as an SDK pack
#include <nethost.h>
#include <coreclr_delegates.h>
#include <hostfxr.h>

#include <iostream>

using string_t = std::basic_string<char_t>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef _WINDOWS

#include <Windows.h>

#define STR(s) L##s
#define CH(c) L##c

namespace
{
  string_t _toString(const Arcane::String& s)
  {
    return string_t((const char_t*)(s.utf16().data()));
  }
  char_t* _duplicate(const char_t* x)
  {
    return ::_wcsdup(x);
  }
  Arcane::String _toArcaneString(const char_t* x)
  {
    using namespace Arcane;
    const UChar* ux = reinterpret_cast<const UChar*>(x);
    size_t slen = wcslen(x);
    Int32 len = CheckedConvert::toInt32(slen);
    ConstArrayView<UChar> buf(len,ux);
    return String(buf);
  }
} // namespace

#else

// UNIX

#include <dlfcn.h>
#include <limits.h>

#define STR(s) s
#define CH(c) c
#define MAX_PATH PATH_MAX
typedef char char_t;
namespace
{
  string_t _toString(const Arcane::String& s)
  {
    if (s.null() || s.empty())
      return string_t();
    return string_t((const char_t*)(s.utf8().data()));
  }
  char_t* _duplicate(const char_t* x)
  {
    return ::strdup(x);
  }
  Arcane::String _toArcaneString(const char_t* x)
  {
    return Arcane::String(Arcane::StringView(x));
  }
} // namespace

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

namespace
{
// Indique si on affiche les informations de debug
int dotnet_verbose = 0;

#ifdef _WINDOWS
using LibHandle = HMODULE;
#else
using LibHandle = void*;
#endif

}

namespace Arcane
{
struct CoreClrLibInfo
{
  LibHandle m_lib_handle = (LibHandle)0;
  bool m_has_valid_lib_handle = false;

  hostfxr_initialize_for_runtime_config_fn init_fptr = nullptr;
  hostfxr_initialize_for_dotnet_command_line_fn init_command_line_fptr = nullptr;
  hostfxr_get_runtime_delegate_fn get_delegate_fptr = nullptr;
  hostfxr_run_app_fn run_app_fptr = nullptr;
  hostfxr_close_fn close_fptr = nullptr;

  void cleanup();
};

}

#define PRINT_FORMAT(level,str,...)             \
  if (dotnet_verbose>=level)\
    std::cout << String::format("[coreclr] " str "\n",__VA_ARGS__);

namespace
{
#if defined(ARCANE_DOTNET_ROOT)
const char* arcane_dotnet_root = ARCANE_DOTNET_ROOT;
#else
const char* arcane_dotnet_root = nullptr;
#endif
Arcane::CoreClrLibInfo lib_info;
// Utile pour conserver la valeur de la variable d'environnement
// DOTNET_ROOT
std::string arcane_dotnet_root_env_variable;

// Globals to hold hostfxr exports

// Forward declarations
bool load_hostfxr(const string_t& assembly_name);

// Load and initialize .NET Core and get desired function pointer for scenario
// (Note: pas utilisé pour l'instant).
load_assembly_and_get_function_pointer_fn
getDotnetLoadAssembly(const String& assembly)
{
  string_t assembly1 = _toString(assembly);
  // Load .NET Core
  void* load_assembly_and_get_function_pointer = nullptr;
  hostfxr_handle cxt = nullptr;

  hostfxr_initialize_parameters params;
  params.size = sizeof(params);
  // TODO: il s'agit du chemin de l'exécutable, pas de l'assembly.
  params.host_path = assembly1.c_str(); //get_path_to_the_host_exe(); // Path to the current executable

  int rc = lib_info.init_fptr(assembly1.c_str(), &params, &cxt);
  if (rc != 0 || cxt == nullptr) {
    auto flags = std::cerr.flags();
    std::cerr << "Init failed: " << std::hex << std::showbase << rc << std::endl;
    std::cerr.setf(flags);
    lib_info.close_fptr(cxt);
    return nullptr;
  }

  // Get the load assembly function pointer
  rc = lib_info.get_delegate_fptr(cxt, hdt_load_assembly_and_get_function_pointer,
                                  &load_assembly_and_get_function_pointer);
  if (rc != 0 || load_assembly_and_get_function_pointer == nullptr)
    std::cerr << "Get delegate failed: " << std::hex << std::showbase << rc << std::endl;

  lib_info.close_fptr(cxt);
  return (load_assembly_and_get_function_pointer_fn)load_assembly_and_get_function_pointer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Pour tester le lancement direct. En cours d'étude.

int
_execDirect(const CommandLineArguments& cmd_args,
            const String& orig_assembly_name)
{
  using const_char_t = const char_t*;

  int argc = *(cmd_args.commandLineArgc());
  //const char** old_argv = (const char**)*(cmd_args.commandLineArgv());

  Arcane::String root_path = Arcane::platform::getFileDirName(orig_assembly_name);
  string_t root_path1 = _toString(root_path);
  string_t orig_assembly_name1 = _toString(orig_assembly_name);
  std::cerr << "ENTERING _execDirect root_path=" << root_path << "\n";

  string_t dotnet_root = _toString(String(arcane_dotnet_root));
  hostfxr_initialize_parameters params;
  params.size = sizeof(params);
  params.host_path = root_path1.c_str();
#ifdef ARCANE_OS_WIN32
  // Sous Windows, '.Net' est installé dans un chemin standard
  // et il ne faut pas spécifier le chemin (cela provoque une erreur
  // d'argument invalide (a vérifier si c'est parce que 'arcane_dotnet_root'
  // n'est pas valide ou s'il ne faut rien spécifier).
  params.dotnet_root = nullptr;
#else
  params.dotnet_root = dotnet_root.c_str();
#endif
  const_char_t* argv = new const_char_t[1];
  char_t* argv0_str = _duplicate((const char_t*)(orig_assembly_name1.c_str()));
  argv[0] = argv0_str;
  std::cerr << "_execDirect argv[0] =" << orig_assembly_name << "\n";
  argc = 1;

  hostfxr_handle host_context_handle;
  int rc = lib_info.init_command_line_fptr(argc, (const char_t**)argv, &params, &host_context_handle);
  std::cerr << "_execDirect init_command_line R = " << rc << "\n";
  if (rc!=0)
    ARCANE_FATAL("Can not initialize runtime RC={0}",rc);

#if TEST
  size_t buffer_used = 0;
  if (hostfxr_get_runtime_property(host_context_handle, "TEST_PROPERTY", nullptr, 0, &buffer_used) == HostApiMissingProperty) {
    hostfxr_set_runtime_property(host_context_handle, "TEST_PROPERTY", "TRUE");
  }
#endif

  std::cerr << "Launching '.Net'\n";
  int r = lib_info.run_app_fptr(host_context_handle);
  std::cerr << "End '.Net': R=" << r << "\n";

  ::free(argv0_str);
  lib_info.close_fptr(host_context_handle);
  return r;
}
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Point d'entrée de la bibliothèque appelé par 'ArcaneMain.cc'
 */

int
_arcaneCoreClrMainInternal(const Arcane::CommandLineArguments& cmd_args,
                           const Arcane::String& orig_assembly_name)
{
  String verbose_str = Arcane::platform::getEnvironmentVariable("ARCANE_DEBUG_DOTNET");
  if (!verbose_str.null())
    dotnet_verbose = 1;

  // Si l'utilisateur a spécifié cette variable, alors on ne surcharge pas
  // les appels avec la version configurée lors de la compilation.
  // Le runtime 'coreclr' se charge d'utiliser cette variable d'environnement.

  String dotnet_root_env = platform::getEnvironmentVariable("DOTNET_ROOT");
  if (!dotnet_root_env.null()){
    arcane_dotnet_root_env_variable = dotnet_root_env.toStdStringView();
    arcane_dotnet_root = arcane_dotnet_root_env_variable.c_str();
  }

  // TODO: trouver un moyen d'utiliser 'cmd_args'
  PRINT_FORMAT(1,"ARCANE_DOTNET_CORECLR_MAIN assembly_name={0}",orig_assembly_name);

  if (orig_assembly_name.empty())
    ARCANE_FATAL("No assembly name");

  string_t orig_assembly_name1 =  _toString(orig_assembly_name);

  String root_path = Arcane::platform::getFileDirName(orig_assembly_name);

  PRINT_FORMAT(1,"ENTERING CORECLR_MAIN root_path={0}",root_path);

  //
  // STEP 1: Load HostFxr and get exported hosting functions
  //
  if (!load_hostfxr(orig_assembly_name1))
    ARCANE_FATAL("Failure: load_hostfxr()");

  const bool do_direct_exec = true;
  if (do_direct_exec)
    return _execDirect(cmd_args, orig_assembly_name);

  // NOTE: Cette partie n'est pas utilisée pour l'instant.

  //
  // STEP 2: Initialize and start the .NET Core runtime
  //
  Arcane::String config_path = Arcane::Directory(root_path).file("Arcane.Main.runtimeconfig.json");
  load_assembly_and_get_function_pointer_fn load_assembly_and_get_function_pointer = nullptr;
  load_assembly_and_get_function_pointer = getDotnetLoadAssembly(config_path);
  assert(load_assembly_and_get_function_pointer != nullptr && "Failure: get_dotnet_load_assembly()");

  //
  // STEP 3: Load managed assembly and get function pointer to a managed method
  //
  //const string_t dotnetlib_path = root_path + STR("Arcane.Main.dll");
  const char_t* dotnet_type = STR("ArcaneMainExec, Arcane.Main");
  const char_t* dotnet_type_method = STR("CoreClrComponentEntryPoint");
  // <SnippetLoadAndGet>
  // Function pointer to managed delegate
  component_entry_point_fn dll_entry_point_func = nullptr;
  // ATTENTION: si on passe 'nullptr' comme 'delegate_type_name', alors 'dotnet_type_method'
  // doit être du type 'int (InpPtr args,int sizeBytes)'.
  int rc = load_assembly_and_get_function_pointer(orig_assembly_name1.c_str(),
                                                  dotnet_type,
                                                  dotnet_type_method,
                                                  nullptr /*delegate_type_name*/,
                                                  nullptr,
                                                  (void**)&dll_entry_point_func);
  // </SnippetLoadAndGet>
  if (rc != 0)
    ARCANE_FATAL("load_assembly_and_get_function_pointer: rc={0}", rc);
  if (!dll_entry_point_func)
    ARCANE_FATAL("Failure: load_assembly_and_get_function_pointer()");

  //
  // STEP 4: Run managed code
  //
  struct lib_args
  {
    const char_t* message;
    int number;
  };
  lib_args args{ STR("from host!"), 0 };

  return dll_entry_point_func(&args, sizeof(args));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Point d'entrée appelé par ArcaneMain
extern "C" ARCANE_EXPORT int
arcane_dotnet_coreclr_main(const Arcane::CommandLineArguments& cmd_args,
                           const Arcane::String& orig_assembly_name)
{
  int ret = 0;
  try{
    ret = _arcaneCoreClrMainInternal(cmd_args,orig_assembly_name);
  }
  catch(const Exception& ex){
    ret = arcanePrintArcaneException(ex,nullptr);
  }
  catch(const std::exception& ex){
    ret = arcanePrintStdException(ex,nullptr);
  }
  catch(...){
    ret = arcanePrintAnyException(nullptr);
  }
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/********************************************************************************************
 * Function used to load and activate .NET Core
 ********************************************************************************************/

namespace
{

// Forward declarations
LibHandle load_library(const char_t*);
void* get_export(LibHandle, const char*);

#ifdef _WINDOWS
LibHandle load_library(const char_t* path)
{
  HMODULE h = ::LoadLibraryW(path);
  if (!h)
    ARCANE_FATAL("Can not load library '{0}'",_toArcaneString(path));
  return h;
}
void free_library(LibHandle h)
{
  FreeLibrary(h);
}
void* get_export(LibHandle h, const char* name)
{
  void* f = ::GetProcAddress(h, name);
  if (!f)
    ARCANE_FATAL("Can not get library symbol '{0}'",name);
  return f;
}
#else
LibHandle load_library(const char_t* path)
{
  void* h = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  if (!h)
    ARCANE_FATAL("Can not load library '{0}' error='{1}'",path,dlerror());
  return h;
}
void free_library(LibHandle h)
{
  ::dlclose(h);
}
void* get_export(LibHandle h, const char* name)
{
  void* f = dlsym(h, name);
  PRINT_FORMAT(1,"get_export name={0} f={1}",name,f);
  if (!f)
    PRINT_FORMAT(0,"Can not get library symbol '{0}'  error='{1}'",name,dlerror());
  return f;
}
#endif

// <SnippetLoadHostFxr>
// Using the nethost library, discover the location of hostfxr and get exports
bool
load_hostfxr(const string_t& assembly_name)
{
  // Si 'coreclr' n'est pas installé dans un chemin standard, il est possible
  // qu'il faille le spécifier. La variable d'environnement DOTNET_ROOT permet
  // de le faire. Si elle n'est pas positionnée, utilise le chemin trouvée
  // lors de la configuration.
  string_t dotnet_root = _toString(String(arcane_dotnet_root));
  get_hostfxr_parameters hostfxr_parameters;
  hostfxr_parameters.size = sizeof(get_hostfxr_parameters);
  hostfxr_parameters.assembly_path = assembly_name.c_str();
  hostfxr_parameters.dotnet_root = dotnet_root.c_str();

  PRINT_FORMAT(1,"Entering load_hostfxr() dotnet_root={0}",arcane_dotnet_root);
  // Pre-allocate a large buffer for the path to hostfxr
  const int BUF_LEN = 12000;
  char_t buffer[BUF_LEN];
  size_t buffer_size = sizeof(buffer) / sizeof(char_t);

  // List of return values for 'get_hostfxr_path' are here:
  // https://github.com/dotnet/runtime/blob/main/docs/design/features/host-error-codes.md
  // Real good value is '0'.
  // Positive values are for warnings
  // Negative valeurs are for errors
  int rc = get_hostfxr_path(buffer, &buffer_size, &hostfxr_parameters);
  PRINT_FORMAT(1,"Return value of 'get_hostfxr_path' = '{0}'",rc);
  if (rc != 0)
    PRINT_FORMAT(0,"Error or warning calling 'get_hostfxr_path' = '{0}'",rc);
  if (rc < 0)
    return false;

  // Load hostfxr and get desired exports
  LibHandle lib = load_library(buffer);
  PRINT_FORMAT(1,"LIB_PTR={0} path={1}",lib,_toArcaneString(buffer));
  lib_info.m_lib_handle = lib;
  lib_info.m_has_valid_lib_handle = true;
  lib_info.init_fptr = (hostfxr_initialize_for_runtime_config_fn)get_export(lib, "hostfxr_initialize_for_runtime_config");
  lib_info.get_delegate_fptr = (hostfxr_get_runtime_delegate_fn)get_export(lib, "hostfxr_get_runtime_delegate");
  lib_info.close_fptr = (hostfxr_close_fn)get_export(lib, "hostfxr_close");
  lib_info.init_command_line_fptr = (hostfxr_initialize_for_dotnet_command_line_fn)get_export(lib, "hostfxr_initialize_for_dotnet_command_line");
  lib_info.run_app_fptr = (hostfxr_run_app_fn)get_export(lib, "hostfxr_run_app");
  return (lib_info.init_fptr && lib_info.get_delegate_fptr && lib_info.close_fptr && lib_info.init_command_line_fptr && lib_info.run_app_fptr);
}

} // namespace

namespace Arcane
{

void CoreClrLibInfo::
cleanup()
{
  if (m_has_valid_lib_handle){
    free_library(m_lib_handle);
    m_lib_handle = (LibHandle)0;
  }
  m_has_valid_lib_handle = false;
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
