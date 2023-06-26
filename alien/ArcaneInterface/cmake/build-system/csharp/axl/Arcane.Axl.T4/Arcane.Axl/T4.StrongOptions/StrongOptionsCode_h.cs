using System;
using System.Linq;

namespace Arcane.Axl
{
  public partial class StrongOptions_h
  {
    public bool WithArcane { get; private set; }

    public string Version { get; private set; }
    
    public OptionHandler Xml { get; private set; }
    
    public Xsd.Base Base { get; private set; }
    
    public StrongOptions_h (Xsd.Module module, string version)
    {
      Version = version;
      WithArcane = true;
      Xml = new OptionHandler (module);
      Base = module;
    }
    
    public StrongOptions_h (Xsd.Service service, string version, bool withArcane)
    {
      Version = version;
      WithArcane = withArcane;
      Xml = new OptionHandler (service);
      Base = service;
    }
  }
}


