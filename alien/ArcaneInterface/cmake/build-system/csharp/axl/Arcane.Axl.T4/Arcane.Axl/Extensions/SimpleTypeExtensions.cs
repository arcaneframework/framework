using System;

namespace Arcane.Axl
{
  public static class SimpleTypeExtensions
  {
    public static string namespaceT = "Arcane";
    public static string exportT = "ARCANE_CORE_EXPORT";

    public static string Export(this Xsd.SimpleType type){
      return exportT;
    }

    public static string Export(){
      return exportT;
    }

    public static string Namespace(this Xsd.SimpleType type){
      return namespaceT;
    }

    public static string Namespace(){
      //  en attendant ArcCore ou pas
      return namespaceT;
    }

    public static string Name(this Xsd.SimpleType type, Boolean isConst = false)
    {
      switch(type){
      case Xsd.SimpleType.ustring:
      case Xsd.SimpleType.@string:
        return "String";
      case Xsd.SimpleType.real:
        return "Real";
      case Xsd.SimpleType.@bool:
        return "Bool";
      case Xsd.SimpleType.integer:
        return "Integer";
      case Xsd.SimpleType.int32:
        return "Int32";
      case Xsd.SimpleType.int64:
        return "Int64";
      case Xsd.SimpleType.real2:
        return "Real2";
      case Xsd.SimpleType.real3:
        return "Real3";
      case Xsd.SimpleType.real2x2:
        return "Real2x2";
      case Xsd.SimpleType.real3x3:
        return "Real3x3";
      case Xsd.SimpleType.string1:
        if(isConst) return "StringConstArrayView";
        else return "StringArray";
      case Xsd.SimpleType.bool1:
        if(isConst) return "BoolConstArrayView";
        else return "BoolArray";
      case Xsd.SimpleType.real1:
        if(isConst) return "RealConstArrayView";
        else return "RealArray";
      case Xsd.SimpleType.real21:
        if(isConst)  return "Real2x2ConstArrayView";
        else return "Real2x2Array";
      case Xsd.SimpleType.real31:
        if(isConst)  return "Real3ConstArrayView";
        else return "Real3Array";
      case Xsd.SimpleType.real2x21:
        if(isConst)  return "Real2x2ConstArrayView";
        else return "Real2x2Array";
      case Xsd.SimpleType.real3x31:
        if(isConst)  return "Real3x3ConstArrayView";
        else return "Real3x3Array";
      case Xsd.SimpleType.integer1:
        if(isConst)  return "IntegerConstArrayView";
        else return "IntegerArray";
      case Xsd.SimpleType.int321:
        if(isConst)  return "Int32ConstArrayView";
        else return "Int32Array";
      case Xsd.SimpleType.int641:
        if(isConst)  return "Int64ConstArrayView";
        else return "Int64Array";
      default:
        throw new TypeUnloadedException (); 
      }
    }

    public static string NameParameter(this Xsd.SimpleType type)
    {
      switch(type){
      case Xsd.SimpleType.ustring:
      case Xsd.SimpleType.@string:
        return "std::string";
      case Xsd.SimpleType.real:
        return "double";
      case Xsd.SimpleType.@bool:
      case Xsd.SimpleType.integer:
      case Xsd.SimpleType.int32:
      case Xsd.SimpleType.int64:
        return "int";
      default:
        throw new TypeUnloadedException (); 
      }
    }


    public static string QualifiedReturnName(this Xsd.SimpleType type, Boolean isConst = false)
    {
      return type.QualifiedName ();
    }

    public static string QualifiedName(this Xsd.SimpleType type)
    {
      if(type == Xsd.SimpleType.@bool)
        return "bool";
      return namespaceT+ "::" + type.Name ();
    }
  }
}

