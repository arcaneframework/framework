using System;
using System.Collections.Generic;

namespace Arcane.Axl
{
  //! Classe utilitaire pour la génération C++
  public static class CppUtils
  {
    class BasicTypeInfo
    {
      public BasicTypeInfo(string name)
      {
        TypeName = name;
      }
      public BasicTypeInfo(string name,string return_name)
      {
        TypeName = name;
        ReturnTypeName = return_name;
      }
      //! Type C++ correspondant
      public string TypeName;
      /*! \brief Type C++ correspondant pour le retour de fonction.
       * Si nul, utilise TypeName.
       */
      public string ReturnTypeName;
    };

    static Dictionary<SimpleOptionInfoType,BasicTypeInfo> m_simple_options;
    static CppUtils ()
    {
      m_simple_options = new Dictionary<SimpleOptionInfoType, BasicTypeInfo>();
      _Add (SimpleOptionInfoType.TypeBool,"Bool");

      _Add(SimpleOptionInfoType.TypeString,"String");
      _Add(SimpleOptionInfoType.TypeReal,"Real");
      _Add(SimpleOptionInfoType.TypeReal2,"Real2");
      _Add(SimpleOptionInfoType.TypeReal3,"Real3");
      _Add(SimpleOptionInfoType.TypeReal2x2,"Real2x2");
      _Add(SimpleOptionInfoType.TypeReal3x3,"Real3x3");
      _Add(SimpleOptionInfoType.TypeInteger,"Integer");
      _Add(SimpleOptionInfoType.TypeInt32,"Int32");
      _Add(SimpleOptionInfoType.TypeInt64,"Int64");
      _Add(SimpleOptionInfoType.TypeStringArray,"StringArray","Arcane::ConstArrayView< Arcane::String >");
      _Add(SimpleOptionInfoType.TypeBoolArray,"BoolArray","Arcane::ConstArrayView<bool>");
      _Add(SimpleOptionInfoType.TypeRealArray,"RealArray","Arcane::ConstArrayView< Arcane::Real >");
      _Add(SimpleOptionInfoType.TypeReal2Array,"Real2Array","Arcane::ConstArrayView< Arcane::Real2 >");
      _Add(SimpleOptionInfoType.TypeReal3Array,"Real3Array","Arcane::ConstArrayView< Arcane::Real3 >");
      _Add(SimpleOptionInfoType.TypeReal2x2Array,"Real2x2Array","Arcane::ConstArrayView< Arcane::Real2x2 >");
      _Add(SimpleOptionInfoType.TypeReal3x3Array,"Real3x3Array","Arcane::ConstArrayView< Arcane::Real3x3 >");
      _Add(SimpleOptionInfoType.TypeIntegerArray,"IntegerArray","Arcane::ConstArrayView< Arcane::Integer >");
      _Add(SimpleOptionInfoType.TypeInt32Array,"Int32Array","Arcane::ConstArrayView< Arcane::Int32 >");
      _Add(SimpleOptionInfoType.TypeInt64Array,"Int64Array","Arcane::ConstArrayView< Arcane::Int64 >");
    }

    //! Nom C++ pour \a st: par exemple, pour 'real' : Real
    static void _Add(SimpleOptionInfoType st,string type_name)
    {
      m_simple_options.Add (st,new BasicTypeInfo(type_name));
    }

    static void _Add(SimpleOptionInfoType st,string type_name,string return_type_name)
    {
      m_simple_options.Add (st,new BasicTypeInfo(type_name,return_type_name));
    }

    static public string BasicTypeName(SimpleOptionInfoType st)
    {
      return m_simple_options[st].TypeName;
    }

    //! Nom qualifié pour \a st: par exemple, pour 'real' : Arcane::Real
    static public string BasicTypeQualifiedName(SimpleOptionInfoType st)
    {
      return "Arcane::"+m_simple_options[st].TypeName;
    }

    //! Nom qualifié pour \a st: par exemple, pour 'real' : Arcane::Real
    static public string ReturnTypeQualifiedName(SimpleOptionInfoType st)
    {
      string return_name = m_simple_options [st].ReturnTypeName;
      if (String.IsNullOrEmpty(return_name))
        return BasicTypeQualifiedName(st);
      return return_name;
    }

    //! Converti un nom de type en nom C++ en tenant compte des namespace
    static public string ConvertType(string name)
    {
      return name.Replace(".","::");
    }

  }
}

