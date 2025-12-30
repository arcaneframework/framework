// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DotNetRuntimeInitialisationInfo.cc                          (C) 2000-2025 */
/*                                                                           */
/* Informations pour l'initialisation du runtime '.Net'.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/DotNetRuntimeInitialisationInfo.h"
#include "arcane/core/internal/DotNetRuntimeInitialisationInfoProperties.h"

#include "arcane/utils/String.h"
#include "arccore/common/internal/Property.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DotNetRuntimeInitialisationInfo::Impl
{
 public:
  bool m_is_using_dotnet_runtime = false;
  String m_main_assembly_name;
  String m_execute_method_name;
  String m_execute_class_name;
  String m_embedded_runtime;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename V> void DotNetRuntimeInitialisationInfoProperties::
_applyPropertyVisitor(V& p)
{
  auto b = p.builder();
  p << b.addString("MainAssemblyName")
        .addDescription("Name of the assembly to load at startup")
        .addCommandLineArgument("DotNetMainAssemblyName")
        .addGetter([](auto a) { return a.x.mainAssemblyName(); })
        .addSetter([](auto a) { a.x.setMainAssemblyName(a.v); });

  p << b.addString("ExecuteMethodName")
        .addDescription("Name of the method to execute")
        .addCommandLineArgument("DotNetExecuteMethodName")
        .addGetter([](auto a) { return a.x.executeMethodName(); })
        .addSetter([](auto a) { a.x.setExecuteMethodName(a.v); });

  p << b.addString("ExecuteClassName")
        .addDescription("Name of the class containing the methode to execute")
        .addCommandLineArgument("DotNetExecuteClassName")
        .addGetter([](auto a) { return a.x.executeClassName(); })
        .addSetter([](auto a) { a.x.setExecuteClassName(a.v); });

  p << b.addString("EmbeddedRuntime")
        .addDescription("Name of the dotnet runtime ('coreclr', 'mono') to use")
        .addCommandLineArgument("DotNetEmbeddedRuntime")
        .addGetter([](auto a) { return a.x.embeddedRuntime(); })
        .addSetter([](auto a) { a.x.setEmbeddedRuntime(a.v); });

  p << b.addBool("UsingDotNet")
        .addDescription("Set/Unset the loading of the '.Net' runtime with 'coreclr'")
        .addCommandLineArgument("UsingDotNet")
        .addGetter([](auto a) { return a.x.isUsingDotNetRuntime(); })
        .addSetter([](auto a) {
          a.x.setIsUsingDotNetRuntime(a.v);
          a.x.setEmbeddedRuntime("coreclr");
        });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DotNetRuntimeInitialisationInfo::
DotNetRuntimeInitialisationInfo()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DotNetRuntimeInitialisationInfo::
DotNetRuntimeInitialisationInfo(const DotNetRuntimeInitialisationInfo& rhs)
: m_p(new Impl(*rhs.m_p))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DotNetRuntimeInitialisationInfo& DotNetRuntimeInitialisationInfo::
operator=(const DotNetRuntimeInitialisationInfo& rhs)
{
  if (&rhs!=this){
    delete m_p;
    m_p = new Impl(*(rhs.m_p));
  }
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DotNetRuntimeInitialisationInfo::
~DotNetRuntimeInitialisationInfo()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DotNetRuntimeInitialisationInfo::
isUsingDotNetRuntime() const
{
  return m_p->m_is_using_dotnet_runtime;
}

void DotNetRuntimeInitialisationInfo::
setIsUsingDotNetRuntime(bool v)
{
  m_p->m_is_using_dotnet_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String DotNetRuntimeInitialisationInfo::
mainAssemblyName() const
{
  return m_p->m_main_assembly_name;
}

void DotNetRuntimeInitialisationInfo::
setMainAssemblyName(StringView v)
{
  m_p->m_main_assembly_name = v;
  if (!v.empty())
    setIsUsingDotNetRuntime(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String DotNetRuntimeInitialisationInfo::
executeClassName() const
{
  return m_p->m_execute_class_name;
}

void DotNetRuntimeInitialisationInfo::
setExecuteClassName(StringView v)
{
  m_p->m_execute_class_name = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String DotNetRuntimeInitialisationInfo::
executeMethodName() const
{
  return m_p->m_execute_method_name;
}

void DotNetRuntimeInitialisationInfo::
setExecuteMethodName(StringView v)
{
  m_p->m_execute_method_name = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String DotNetRuntimeInitialisationInfo::
embeddedRuntime() const
{
  return m_p->m_embedded_runtime;
}

void DotNetRuntimeInitialisationInfo::
setEmbeddedRuntime(StringView v)
{
  m_p->m_embedded_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_PROPERTY_CLASS(DotNetRuntimeInitialisationInfoProperties,());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

