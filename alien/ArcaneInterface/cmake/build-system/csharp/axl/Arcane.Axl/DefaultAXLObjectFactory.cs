using System;
using System.Xml;

namespace Arcane.Axl
{
  /**
   * \brief Fabrique par defaut pour les objets des AXL.
   */
  public class DefaultAXLObjectFactory : IAXLObjectFactory
  {
    public DefaultAXLObjectFactory ()
    {
    }

    public VariableInfo CreateVariableInfo(XmlElement variable_elem)
    {
      return new VariableInfo(variable_elem);
    }
  }
}

