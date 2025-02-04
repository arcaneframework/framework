// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommandLineArguments.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Arguments de la ligne de commande.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/String.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ParameterList.h"

#include <atomic>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CommandLineArguments::Impl
{
 public:
  /*!
   * \brief Paramètres de la ligne de commande.
   *
   * Ils sont récupérés via l'option '-A' de la ligne de commande
   * et sont de la forme -A,x=y,a=b.
   */
  class NameValuePair
  {
   public:
    NameValuePair(const String& n,const String& v) : name(n), value(v){}
    String name;
    String value;
  };
 public:
  Impl(int* argc,char*** argv)
  : m_nb_ref(0)
  , m_args()
  , m_argc(argc)
  , m_argv(argv)
  , m_need_destroy(false)
  , m_need_help(false)
  {
  }

  Impl(const StringList& aargs)
  : m_nb_ref(0)
  , m_args(aargs)
  , m_argc(nullptr)
  , m_argv(nullptr)
  , m_need_destroy(true)
  , m_need_help(false)
  {
    Integer nb_arg = aargs.count();
    m_argc_orig = new int;
    m_argc = m_argc_orig;
    *m_argc = nb_arg+1;

    m_argv_orig = new char**;
    char*** argv = m_argv_orig;
    *argv = new char*[nb_arg+1];
    m_argv0 = ::strdup("arcane");
    (*argv)[0] = m_argv0;
    for(Integer i=0; i<nb_arg; ++i )
      (*argv)[i+1] = (char*)m_args[i].localstr();
    m_argv = argv;
  }

  Impl()
  : m_nb_ref(0)
  , m_args()
  , m_argc(nullptr)
  , m_argv(nullptr)
  , m_need_destroy(true)
  , m_need_help(false)
  {
    m_argc_orig = new int;
    m_argc = m_argc_orig;
    *m_argc = 1;

    m_argv_orig = new char**;
    char*** argv = m_argv_orig;
    *argv = new char*[1];
    m_argv0 = ::strdup("arcane");
    (*argv)[0] = m_argv0;
    m_argv = argv;
  }
  ~Impl()
  {
    if (m_need_destroy){
      delete m_argc_orig;
      if (m_argv_orig)
        delete[] (*m_argv_orig);
      delete m_argv_orig;
      ::free(m_argv0);
    }
  }
 public:
  void addReference() { ++m_nb_ref; }
  void removeReference()
  {
    // Décrémente et retourne la valeur d'avant.
    // Si elle vaut 1, cela signifie qu'on n'a plus de références
    // sur l'objet et qu'il faut le détruire.
    Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
    if (v==1)
      delete this;
  }
  void parseParameters(const CommandLineArguments& command_line_args)
  {
    // On ne récupère que les arguments du style:
    //   -A,x=b,y=c
    StringList args;
    command_line_args.fillArgs(args);
    if (args.count() == 1) {
      m_need_help = true;
      return;
    }
    for (Integer i = 0, n = args.count(); i < n; ++i) {
      String arg = args[i];
      if (arg.startsWith("-h") || arg.startsWith("--help")) {
        m_need_help = true;
        // TODO AH : Voir pour faire une aide : "-h=module".
        continue;
      }
      if (!arg.startsWith("-A,"))
        continue;
      String arg_value = arg.substring(3);
      if (arg_value.null() || arg_value.empty())
        continue;
      UniqueArray<String> values;
      arg_value.split(values,',');
      for( const auto& x : values ){
        m_parameter_list.addParameterLine(x);
      }
    }
  }
  String getParameter(const String& param_name)
  {
    return m_parameter_list.getParameterOrNull(param_name);
  }

  void fillParameters(StringList& param_names,StringList& values) const
  {
    m_parameter_list.fillParameters(param_names,values);
  }

  bool needHelp() const
  {
    return m_need_help;
  }

 public:
  std::atomic<Int32> m_nb_ref;
  StringList m_args;
  int* m_argc; //!< Nombre d'arguments de la ligne de commande
  char*** m_argv; //!< Tableau des arguments de la ligne de commande
  int* m_argc_orig = nullptr;
  char*** m_argv_orig = nullptr;
  char* m_argv0 = nullptr;
  bool m_need_destroy;
  bool m_need_help;
  ParameterList m_parameter_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommandLineArguments::
CommandLineArguments(int* argc,char*** argv)
: m_p(new Impl(argc,argv))
{
  m_p->parseParameters(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommandLineArguments::
CommandLineArguments()
: m_p(new Impl())
{
  m_p->parseParameters(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommandLineArguments::
CommandLineArguments(const StringList& aargs)
: m_p(new Impl(aargs))
{
  m_p->parseParameters(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommandLineArguments::
CommandLineArguments(const CommandLineArguments& rhs)
: m_p(rhs.m_p)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommandLineArguments& CommandLineArguments::
operator=(const CommandLineArguments& rhs)
{
  m_p = rhs.m_p;
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommandLineArguments::
~CommandLineArguments()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int* CommandLineArguments::
commandLineArgc() const
{
  return m_p->m_argc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

char*** CommandLineArguments::
commandLineArgv() const
{
  return m_p->m_argv;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CommandLineArguments::
fillArgs(StringList& aargs) const
{
  int nargc = *m_p->m_argc;
  char** nargv = *m_p->m_argv;
  aargs.resize(nargc);
  for( int i=0; i<nargc; ++i )
    aargs[i] = nargv[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CommandLineArguments::
getParameter(const String& param_name) const
{
  return m_p->getParameter(param_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CommandLineArguments::
addParameterLine(const String& line)
{
  m_p->m_parameter_list.addParameterLine(line);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CommandLineArguments::
fillParameters(StringList& param_names,StringList& values) const
{
  m_p->fillParameters(param_names,values);
}
 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ParameterList& CommandLineArguments::
parameters() const
{
  return m_p->m_parameter_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CommandLineArguments::
needHelp() const
{
  return m_p->needHelp();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

