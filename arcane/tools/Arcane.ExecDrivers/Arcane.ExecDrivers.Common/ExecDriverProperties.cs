//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System.Collections.Generic;

namespace Arcane.ExecDrivers.Common
{
  public class ExecDriverProperties
  {
    public string ExecName { get; internal set; }
    public int NbProc { get; internal set; }
    public int NbIteration { get; internal set; }
    public int NbContinue { get; internal set; }
    public int NbSharedMemorySubDomain { get; internal set; }
    public int NbTaskPerProcess { get; internal set; }
    public int NbReplication { get; internal set; }
    public bool UseTotalview { get; set; }
    public bool UseDdt { get; set; }
    public string DirectExecMethod { get; set; }
    public List<string> MpiLauncherArgs;
    public string MpiLauncher;
    public ExecDriverProperties()
    {
      MpiLauncherArgs = new List<string>();
      ExecName = "arcane_tests_exec";
    }
  }
}
