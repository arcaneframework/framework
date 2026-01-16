// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ApplicationInfo.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Informations sur une application.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/String.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/List.h"
#include "arccore/common/internal/Property.h"
#include "arcane/utils/internal/ApplicationInfoProperties.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ApplicationInfoPrivate
{
 public:
  ApplicationInfoPrivate(const CommandLineArguments& args,const String& name,
                         const VersionInfo& version)
  : m_command_line_args(args), m_version(version), m_is_debug(false), m_application_name(name)
  {
  }
 public:
  CommandLineArguments m_command_line_args;
  VersionInfo m_version; //!< Numéro de version
  bool m_is_debug; //!< \a true s'il s'agit d'une version debug.
  String m_version_date; //!< Date de la version.
  String m_application_name; //!< Nom de l'application
  String m_code_name; //!< Nom du code
  String m_target_full_name; //!< Nom complet de la cible
  String m_osname; //!< Nom de l'OS
  String m_data_os_dir; //!< Répertoire des fichiers config dépendant OS
  String m_data_dir; //!< Répertoire des fichiers de config communs
  StringList m_args;
  StringList m_dynamic_libraries_name;
  UniqueArray<std::byte> m_runtime_config_file_content;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationInfo::
ApplicationInfo()
: m_argc(nullptr)
, m_argv(nullptr)
, m_p(new ApplicationInfoPrivate(CommandLineArguments(StringList()),"Arcane",VersionInfo(1,0,0)))
{
  _init("Arcane");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationInfo::
ApplicationInfo(int* argc,char*** argv,const String& name,const VersionInfo& aversion)
: m_argc(nullptr)
, m_argv(nullptr)
, m_p(new ApplicationInfoPrivate(CommandLineArguments(argc,argv),name,aversion))
{
  _init(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationInfo::
ApplicationInfo(const StringList& aargs,const String& name,
                const VersionInfo& aversion)
: m_argc(nullptr)
, m_argv(nullptr)
, m_p(new ApplicationInfoPrivate(CommandLineArguments(aargs),name,aversion))
{
  _init(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationInfo::
ApplicationInfo(const CommandLineArguments& aargs,const String& name,
                const VersionInfo& aversion)
: m_argc(nullptr)
, m_argv(nullptr)
, m_p(new ApplicationInfoPrivate(aargs,name,aversion))
{
  _init(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationInfo::
ApplicationInfo(const ApplicationInfo& rhs)
: m_argc(nullptr)
, m_argv(nullptr)
, m_p(new ApplicationInfoPrivate(*rhs.m_p))
{
  _setArgs();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationInfo& ApplicationInfo::
operator=(const ApplicationInfo& rhs)
{
  if (&rhs!=this){
    auto old_p = m_p;
    m_p = new ApplicationInfoPrivate(*rhs.m_p);
    delete old_p;
    _setArgs();
  }
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationInfo::
~ApplicationInfo()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
_setArgs()
{
  m_argc = m_p->m_command_line_args.commandLineArgc();
  m_argv = m_p->m_command_line_args.commandLineArgv();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
_init(const String& name)
{
  _setArgs();

  m_p->m_version_date = "0";
  m_p->m_data_os_dir = String();
  m_p->m_data_dir = String();
  
  {
    String s = platform::getEnvironmentVariable("ARCANE_USER_CONFIG_FILE_NAME");
    if (!s.null())
      m_p->m_code_name = s;
    else
      m_p->m_code_name = name;
  }

  {
    String s = platform::getEnvironmentVariable("ARCANE_SHARE_PATH");
    if (!s.null())
      m_p->m_data_dir = s;
    else{
      s = platform::getEnvironmentVariable("STDENV_PATH_SHR");
      if (!s.null())
        m_p->m_data_dir = s;
    }
  }
  {
    String s = platform::getEnvironmentVariable("ARCANE_LIB_PATH");
    if (!s.null())
      m_p->m_data_os_dir = s;
    else{
      String s = platform::getEnvironmentVariable("STDENV_PATH_LIB");
      if (!s.null())
        m_p->m_data_os_dir = s;
    }
  }

  if (dataDir().null() || dataOsDir().null()){
    // Considère que les infos partagées et les libs sont
    // au même endroit qui est celui de l'exécutable.
    String exe_full_path = platform::getExeFullPath();
    String exe_path = platform::getFileDirName(exe_full_path);
    if (dataDir().null())
      setDataDir(exe_path);
    if (dataOsDir().null())
      setDataOsDir(exe_path);
  }

  m_p->m_target_full_name = "TargetUnknown";

#ifdef ARCANE_DEBUG
  m_p->m_is_debug = true;
#else
  m_p->m_is_debug = false;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& ApplicationInfo::
applicationName() const
{
  return m_p->m_application_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VersionInfo& ApplicationInfo::
codeVersion() const
{
  return m_p->m_version;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& ApplicationInfo::
dataOsDir() const
{
  return m_p->m_data_os_dir;
}

void ApplicationInfo::
setDataOsDir(const String& v)
{
  m_p->m_data_os_dir = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& ApplicationInfo::
dataDir() const
{
  return m_p->m_data_dir;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
setDataDir(const String& v)
{
  m_p->m_data_dir = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ApplicationInfo::
versionMajor() const
{
  return m_p->m_version.versionMajor();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ApplicationInfo::
versionMinor() const
{
  return m_p->m_version.versionMinor();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ApplicationInfo::
versionPatch() const
{
  return m_p->m_version.versionPatch();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ApplicationInfo::
isDebug() const
{
  return m_p->m_is_debug;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& ApplicationInfo::
codeName() const
{
  return m_p->m_code_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& ApplicationInfo::
targetFullName() const
{
  return m_p->m_target_full_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int* ApplicationInfo::
commandLineArgc() const
{
  return m_argc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

char*** ApplicationInfo::
commandLineArgv() const
{
  return m_argv;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
args(StringList& aargs) const
{
  m_p->m_command_line_args.fillArgs(aargs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
addDynamicLibrary(const String& lib_name)
{
  m_p->m_dynamic_libraries_name.add(lib_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringCollection ApplicationInfo::
dynamicLibrariesName() const
{
  return m_p->m_dynamic_libraries_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const CommandLineArguments& ApplicationInfo::
commandLineArguments() const
{
  return m_p->m_command_line_args;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
setApplicationName(const String& v)
{
  m_p->m_application_name = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
setCodeVersion(const VersionInfo& version)
{
  m_p->m_version = version;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
setCodeName(const String& code_name)
{
  m_p->m_code_name = code_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
setCommandLineArguments(const CommandLineArguments& aargs)
{
  m_p->m_command_line_args = aargs;
  _setArgs();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
setIsDebug(bool v)
{
  m_p->m_is_debug = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
setRuntimeConfigFileContent(ByteConstSpan content)
{
  m_p->m_runtime_config_file_content = content;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ByteConstSpan ApplicationInfo::
runtimeConfigFileContent() const
{
  return m_p->m_runtime_config_file_content;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ApplicationInfo::
addParameterLine(const String& line)
{
  m_p->m_command_line_args.addParameterLine(line);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename V> void ApplicationInfoProperties::
_applyPropertyVisitor(V& p)
{
  auto b = p.builder();

  p << b.addString("CodeName")
        .addDescription("Name of the code")
        .addGetter([](auto a) { return a.x.codeName(); })
        .addSetter([](auto a) { a.x.setCodeName(a.v); });

  p << b.addString("DataDir")
        .addDescription("Directory containing os independant files")
        .addGetter([](auto a) { return a.x.dataDir(); })
        .addSetter([](auto a) { a.x.setDataDir(a.v); });

  p << b.addString("DataOsDir")
        .addDescription("Directory containing os dependant files")
        .addGetter([](auto a) { return a.x.dataOsDir(); })
        .addSetter([](auto a) { a.x.setDataOsDir(a.v); });

  p << b.addBool("Debug")
        .addDescription("Indicate if debug mode is active")
        .addGetter([](auto a) { return a.x.isDebug(); })
        .addSetter([](auto a) { a.x.setIsDebug(a.v); });

  p << b.addString("CodeVersion")
        .addDescription("Version (x.y.z) of the code")
        .addSetter([](auto a) { a.x.setCodeVersion(VersionInfo(a.v)); })
        .addGetter([](auto a) { return a.x.codeVersion().versionAsString(); });

  p << b.addStringList("DynamicLibraries")
        .addDescription("Dynamic libraries to load at startup")
        .addSetter([](auto a)
                   {
                     for(String s : a.v)
                       a.x.addDynamicLibrary(s);
                   })
        .addGetter([](auto a) { return a.x.dynamicLibrariesName(); });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_PROPERTY_CLASS(ApplicationInfoProperties,());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

