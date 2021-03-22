using System;

namespace Arcane.Axl
{
  public static class PropertyExtensions
  {
    public static string Name(this Xsd.Property property)
    {
      switch(property){
      case Xsd.Property.none:
        return "PNone";
      case Xsd.Property.autoloadbegin:
        return "PAutoLoadBegin";
      case Xsd.Property.autoloadend:
        return "PAutoLoadEnd";
      default:
        throw new TypeUnloadedException ();
      }
    }
  }
}

