using System;

namespace Arcane.Axl
{
  public partial class ServiceT4Standard
  {
    private Xsd.Service Service { get; set; }

    public string Path { get; private set; }
    public string Version { get; private set; }
    public OptionHandler Xml { get; private set; }
    
    public ServiceT4Standard (Xsd.Service service, string path, string version)
    {
      Version = version;
      Service  = service;
      // add test service type CaseOption
      Path = path;
      Xml = new OptionHandler (service);
    }
  }
}

