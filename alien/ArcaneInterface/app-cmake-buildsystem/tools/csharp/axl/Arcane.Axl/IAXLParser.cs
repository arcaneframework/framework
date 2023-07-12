using System;

namespace Arcane.Axl
{
  /** Interface d'un parseur de fichier AXL. */
  public interface IAXLParser
  {
    IAXLObjectFactory Factory { get; set; }

    Option ParseSubElements(OptionBuildInfo build_info);
  }
}

