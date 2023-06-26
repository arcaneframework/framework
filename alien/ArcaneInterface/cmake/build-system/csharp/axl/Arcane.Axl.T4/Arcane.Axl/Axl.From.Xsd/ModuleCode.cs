using System;
using System.Collections.Generic;
using System.Linq;

namespace Arcane.Axl.Xsd
{
  public partial class Module
  {
    public Module ()
    {
    }

    public IEnumerable<entrypointsEntrypoint> EntryPoints {
      get {
        if (entrypoints == null)
          return new List<entrypointsEntrypoint> ();
        else
          return new List<entrypointsEntrypoint> (entrypoints);
      }
    }

    public IEnumerable<Xsd.variablesVariable> Variables { 
      get {
        if (variables == null)
          return new List<Xsd.variablesVariable> ();
        else
          return new List<Xsd.variablesVariable> (variables);
      }
    }
  }
}

