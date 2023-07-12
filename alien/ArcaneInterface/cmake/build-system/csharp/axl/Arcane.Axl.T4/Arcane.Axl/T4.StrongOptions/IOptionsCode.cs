using System;
using System.Linq;

namespace Arcane.Axl
{
  public partial class IOptions
  {
    public bool WithArcane { get; private set; }

    public string Version { get; private set; }
    
    public OptionHandler Xml { get; private set; }
    
    public Xsd.Base Base { get; private set; }
    
    public IOptions (Xsd.Module module, string version)
    {
      Version = version;
      Xml = new OptionHandler (module);
      Base = module;
    }
    
    public IOptions (Xsd.Service service, string version, bool withArcane)
    {
      Version = version;
      Xml = new OptionHandler (service);
      Base = service;
      WithArcane = withArcane;
    }
  }
}



