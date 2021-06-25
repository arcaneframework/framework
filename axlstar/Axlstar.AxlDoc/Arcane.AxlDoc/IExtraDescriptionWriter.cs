using System.Collections.Generic;
using System.IO;
using System.Xml.Linq;
using Arcane.Axl;

namespace Arcane.AxlDoc
{
  public interface IExtraDescriptionWriter
  {
    void writeDescription(Option option, IEnumerable<XElement> description_nodes,TextWriter stream);
  }
}