using System;
using System.Linq;

namespace Arcane.Axl
{
  public partial class OptionsUtils
  {
   
    public OptionHandler Xml { get; private set; }
    
    public Xsd.Base Base { get; private set; }
    
    public OptionsUtils (Xsd.Service service)
    {
      Xml = new OptionHandler (service);
      Base = service;
    }
  }
}

