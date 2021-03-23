using System;

namespace Arcane
{
  public class ModuleAttribute : Attribute
  {
    //! Nom du module
    public string Name { set; get; }
    
    //! Version du module
    public string Version { set; get; }

    public ModuleAttribute(string name,string version)
    {
      Name = name;
      Version = version;
    }

    public ModuleAttribute(string name)
    {
      Name = name;
      Version = "0.0.0";
    }
  }
}
