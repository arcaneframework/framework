using System;
using System.Collections.Generic;

namespace Arcane.Axl.Xsd
{
  public partial class Base
  {
    public Base ()
    {
    }

    public string Name { get { return name1; } } 
  
    public IEnumerable<name> Names {
      get {
        if (name == null)
          return new List<name> ();
        else
          return new List<name> (name);
      }
    }
  }
}

