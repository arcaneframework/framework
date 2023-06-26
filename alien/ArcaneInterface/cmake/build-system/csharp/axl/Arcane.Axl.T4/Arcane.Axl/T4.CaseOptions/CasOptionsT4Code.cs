using System;
using System.Linq;
using System.Collections.Generic;
namespace Arcane.Axl
{
  public partial class CaseOptionsT4
  {
    public string Version { get; private set; }

    public OptionHandler Xml { get; private set; }

    public Xsd.Base Base { get; private set; }

    public bool ComplexContainsServiceInstance { get; private set; }

    public CaseOptionsT4 (Xsd.Module module, string version)
    {
      Version = version;
      Xml = new OptionHandler (module);
      Base = module;
      ComplexContainsServiceInstance = false;
    }

    public CaseOptionsT4 (Xsd.Service service, string version)
    {
      Version = version;
      Xml = new OptionHandler (service);
      Base = service;
      ComplexContainsServiceInstance = false;
      foreach(var complex in Xml.FlatteningComplex.Where( p => !p.IsRef)){
        foreach(var e in complex.Xml.ServiceInstance){
          ComplexContainsServiceInstance = true;
          break;
        }
      }
    }
  }
}

