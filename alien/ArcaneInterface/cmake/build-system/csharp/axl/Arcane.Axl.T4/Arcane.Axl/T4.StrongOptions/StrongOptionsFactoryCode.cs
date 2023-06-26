using System;
using System.Linq;

namespace Arcane.Axl
{
  public partial class StrongOptionsFactory
  {
    public bool WithArcane { get; private set; }
    
    public OptionHandler Xml { get; private set; }
    
    public Xsd.Base Base { get; private set; }

    public StrongOptionsFactory (Xsd.Service service, bool withArcane)
    {
      Xml = new OptionHandler (service);
      Base = service;
      WithArcane = withArcane;
    }
  }
}



