//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Xunit;
using System.IO;

namespace Axlstar.Axl2cc.Tests
{
  public class Test
  {
    [Fact()]
    public void TestCase1()
    {
      _ApplyAxl2cc("CaseOptionsTester.axl");
    }
    [Fact()]
    public void TestCase2()
    {
      _ApplyAxl2cc("Ensight7PostProcessor.axl");
    }

    void _ApplyAxl2cc(string axl_file_name)
    {
      var v = new Axlstar.Axl.Generator();
      string file_path = Path.Combine("axlfiles", axl_file_name);
      string[] args = new string[] { file_path };
      int r = v.Execute(args);
      if (r != 0)
        throw new ApplicationException("Error during axl2cc");
    }
  }
}
