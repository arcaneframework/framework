using System;

namespace Arcane.Axl
{
  public partial class ServiceT4CaseAndStrong
  {
    private Xsd.Service Service { get; set; }

    public string Path { get; private set; }
    public string Version { get; private set; }
    public bool WithMesh { get; private set; }
    public OptionHandler Xml { get; private set; }
    
    public ServiceT4CaseAndStrong (Xsd.Service service, string path, string version, bool withMesh)
    {
      Version = version;
      Service  = service;
      // add test service type CaseOption
      Path = path;
      // should be define in each axl File
      WithMesh = withMesh;
      Xml = new OptionHandler (service);
    }
  }
}

