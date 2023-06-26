using System;

namespace Arcane.Axl
{
  public partial class ServiceT4 
  {
    private Xsd.Service Service { get; set; }

    public string Path { get; private set; }
    public string Version { get; private set; }
    public bool WithStrongOption { get; private set; }
    public bool WithoutMesh { get; private set; }
    public OptionHandler Xml { get; private set; }
    
    public ServiceT4 (Xsd.Service service, string path, string version, bool with_strong_option, 
                      bool with_case_option, bool without_mesh, bool with_arcane)
    {
      Version = version;
      Service  = service;
      Path = path;
      WithStrongOption = with_strong_option;
      WithCaseOption = with_case_option;
      WithoutMesh = without_mesh;
      WithArcane = with_arcane;
      Xml = new OptionHandler (service);
    }
  }
}

