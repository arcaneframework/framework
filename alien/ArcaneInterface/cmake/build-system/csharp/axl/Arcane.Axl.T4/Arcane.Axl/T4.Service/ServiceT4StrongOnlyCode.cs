using System;

namespace Arcane.Axl
{
  public partial class ServiceT4StrongOnly 
  {
    private Xsd.Service Service { get; set; }

    public bool WithMesh { get; private set; }
    public bool WithArcane { get; private set; }
    public string Path { get; private set; }
    public string Version { get; private set; }
    public OptionHandler Xml { get; private set; }
    
    public ServiceT4StrongOnly (Xsd.Service service, string path, string version, bool withArcane, bool withMesh)
    {
      // add test service type CaseOption
      Version = version;
      Service  = service;
      Path = path;
      WithMesh = withMesh;
      WithArcane = withArcane;
      Xml = new OptionHandler (service);
    }
  }
}

