using System;
using System.Collections.Generic;
using System.Linq;

namespace LawCompiler
{
  public static class Extensions
  {
    public static string Name(this PropertyDim dim)
    {
      if (dim == PropertyDim.vectorial) 
        return "Vectorial";
      else 
        return "Scalar";
    }
    
    public static string Name(this PropertyType type)
    {
      if (type == PropertyType.integer) 
        return "Integer";
      else 
        return "Real";
    }
  }

  public partial class Property
  {
    public string DataType {
      get {
        if (dim == PropertyDim.scalar)
          return "Arcane::" + type.Name ();
        else
          return "Arcane::" + type.Name () + "SharedArray";
      }
    }

    public string InSignatureType {
      get {
        if (dim == PropertyDim.scalar)
          return "Arcane::" + type.Name ();
        else
          return "Arcane::" + type.Name () + "ConstArrayView";
      }
    }

    public string OutSignatureType {
      get {
        if (dim == PropertyDim.scalar)
          return "Arcane::" + type.Name () + "&";
        else 
          return "Arcane::" + type.Name () + "ArrayView";
      }
    }

    public string outSignatureType (Property p) {
      if (dim == PropertyDim.scalar) {
        if(p.dim == PropertyDim.scalar)
          return "Arcane::" + type.Name () + "&";
        else
          return "Arcane::" + type.Name () + "ArrayView";
      }
      else { 
        if(p.dim == PropertyDim.scalar) 
          return "Arcane::" + type.Name () + "ArrayView";
        else 
          return "Arcane::" + type.Name () + "Array2View";
      }
    }

    public string Type
    {
      get {
        if(dim == PropertyDim.multiscalar) 
          return "Arcane::UniqueArray<Law::Scalar" + type.Name() + "Property>";
        else
          return "Law::" + dim.Name() + type.Name() + "Property";
      }
    }

    public Property ()
    {
    }
  }
}

