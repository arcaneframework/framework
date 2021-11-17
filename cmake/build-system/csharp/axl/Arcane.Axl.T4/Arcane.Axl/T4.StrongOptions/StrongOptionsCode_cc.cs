using System;
using System.Linq;

namespace Arcane.Axl
{
  public partial class StrongOptions_cc
  {
    public string Version { get; private set; }
    
    public OptionHandler Xml { get; private set; }
    
    public Xsd.Base Base { get; private set; }
    
    public StrongOptions_cc (Xsd.Module module, string version)
    {
      Version = version;
      Xml = new OptionHandler (module);
      Base = module;
    }
    
    public StrongOptions_cc (Xsd.Service service, string version)
    {
      Version = version;
      Xml = new OptionHandler (service);
      Base = service;
    }
  }
}


