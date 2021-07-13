using System.Collections.Generic;
using System.IO;
using System.Xml.Linq;


namespace Arcane.Axl
{
  public interface IExtraDescriptionWriter
  {
    void writeDescription(IEnumerable<XElement> description_nodes,TextWriter stream);
  }

}