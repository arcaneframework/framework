using System;
using System.Collections.Generic;

namespace Arcane.Axl
{
  public partial class ModuleT4 
  {
    private Xsd.Module Module { get; set; }

    public string Version { get; private set; }
    public OptionHandler Xml { get; private set; }

    public ModuleT4 (Xsd.Module module, string version)
    {
      Version = version;
      Module  = module;
      Xml = new OptionHandler (module);
    }
  }
}

