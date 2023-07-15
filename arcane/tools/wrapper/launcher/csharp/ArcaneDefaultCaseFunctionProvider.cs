//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Arcane;
using System.Reflection;

namespace Arcane.DefaultService
{
  [Arcane.Service("ArcaneDefaultDotNetCaseFunctionProvider",typeof(Arcane.ICaseFunctionDotNetProvider))]
  public class DotNetDefaultCaseFunctionProvider : Arcane.ICaseFunctionDotNetProvider_WrapperService
  {
    public DotNetDefaultCaseFunctionProvider(ServiceBuildInfo bi) : base(bi)
    {
    }

    public override void RegisterCaseFunctions(ICaseMng cm,string assembly_name,
                                               string class_name)
    {
      TraceAccessor trace = new TraceAccessor(cm.TraceMng());
      trace.Info("REGISTER C# CaseFunctions: assembly_name={0} class_name={1}",assembly_name,class_name);
      Arcane.CaseFunctionLoader.LoadCaseFunction(cm,assembly_name,class_name);
    }
  }
}
