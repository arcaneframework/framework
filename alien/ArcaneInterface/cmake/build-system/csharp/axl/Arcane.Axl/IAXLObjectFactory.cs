using System;
using System.Xml;

namespace Arcane.Axl
{
  /**
   * \brief Interface pour les fabriques des objets disponibles dans les fichiers AXL.
   * 
   * Cette interface permet de fabriquer par exemple les objets de type VariableInfo.
   */
  public interface IAXLObjectFactory
  {
    VariableInfo CreateVariableInfo(XmlElement variable_node);
  }
}

