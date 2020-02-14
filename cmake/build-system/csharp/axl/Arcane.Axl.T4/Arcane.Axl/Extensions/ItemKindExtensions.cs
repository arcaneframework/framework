using System;

namespace Arcane.Axl
{
  public static class ItemKindExtensions
  {
    public static string Name(this Xsd.ItemKind type)
    {
      switch(type){
      case Xsd.ItemKind.cell:
        return "Cell";
      case Xsd.ItemKind.dualnode:
        return "DualNode";
      case Xsd.ItemKind.edge:
        return "Edge";
      case Xsd.ItemKind.face:
        return "Face";
      case Xsd.ItemKind.link:
        return "Link";
      case Xsd.ItemKind.node:
        return "Node";
      default:
        throw new TypeUnloadedException (); 
      }
    }
  }
}

