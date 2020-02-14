using System;

namespace Arcane.Axl
{
  public static class DataTypeExtensions
  {
    public static string Name(this Xsd.DataType type)
    {
      switch(type){
      case Xsd.DataType.@bool:
        return "Bool";
      case Xsd.DataType.int32:
        return "Int32";
      case Xsd.DataType.int64:
        return "Int64";
      case Xsd.DataType.integer:
        return "Integer";
      case Xsd.DataType.real:
        return "Real";
      case Xsd.DataType.real2:
        return "Real2";
      case Xsd.DataType.real2x2:
        return "Real2x2";
      case Xsd.DataType.real3:
        return "Real3";
      case Xsd.DataType.real3x3:
        return "Real3x3";
      case Xsd.DataType.@string:
        return "String";
      default:
        throw new TypeUnloadedException (); 
      }
    }
  }
}

