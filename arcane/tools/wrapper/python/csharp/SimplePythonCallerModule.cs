//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using Arcane;
using System;
using Python.Runtime;

[Arcane.Module("SimplePythonCallerModule","0.0.1")]
class SimplePythonCallerModule
: Arcane.BasicModule
{
  public SimplePythonCallerModule(ModuleBuildInfo infos) : base(infos)
  {
    Console.WriteLine("Create SimplePythonCallerModule");
    _AddEntryPoint("Main1",MyEntryPoint,IEntryPoint.WComputeLoop,0);
    _AddEntryPoint("OnExit",OnExit,IEntryPoint.WExit,0);
    PythonEngine.Initialize();
  }

  void MyEntryPoint()
  {
    Console.WriteLine("Test calling Python");
    
    int c = -1;
    using (Py.GIL())
    using (var scope = Py.CreateScope())
    {
      int p1 = 25;
      scope.Set("my_value",p1);
      //scope.Set(parameterName, parameter.ToPython());
      scope.Exec("a=my_value\nb=5\nc=a+b\nprint(c)\n");
      c = scope.Get<int>("c");
    }
    Console.WriteLine("C={0}",c);
    Console.WriteLine("End setup python");
  }

  void OnExit()
  {
    Console.WriteLine("Exiting python");
    PythonEngine.Shutdown();
  }
}
