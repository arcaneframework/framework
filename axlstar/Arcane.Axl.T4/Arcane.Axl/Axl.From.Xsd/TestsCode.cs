using System;
using System.Collections.Generic;

namespace Arcane.Axl.Xsd
{
  public partial class tests
  {
    public tests ()
    {
    }

    public IEnumerable<testsTest> Tests { 
      get { 
        if (test == null) {
          return new List<testsTest> ();
        } else {
          return test; 
        }
      }
    }
  }
}

