/*---------------------------------------------------------------------------*/
/* SimpleOptionInfo.cs                                         (C) 2000-2009 */
/*                                                                           */
/* Classe stockant les informations de l'élément XML "simple".               */
/*---------------------------------------------------------------------------*/
using System.Xml;
using System;

namespace Arcane.Axl
{
  /// <summary>
  /// Type de donnée d'une option simple.
  /// </summary>
  public enum SimpleOptionInfoType
  {
    TypeString,
    TypeBool,
    TypeReal,
    TypeInteger,
    TypeInt32,
    TypeInt64,
    TypeReal2,
    TypeReal2x2,
    TypeReal3,
    TypeReal3x3,
    TypeStringArray,
    TypeBoolArray,
    TypeRealArray,
    TypeIntegerArray,
    TypeInt32Array,
    TypeInt64Array,
    TypeReal2Array,
    TypeReal2x2Array,
    TypeReal3Array,
    TypeReal3x3Array,
  };

  /// <summary>
  /// Classe stockant les informations de l'élément XML "simple". 
  /// </summary>
  public class SimpleOptionInfo : Option
  {
    public SimpleOptionInfo(OptionBuildInfo build_info)
    : base(build_info)
    {
      XmlElement node = build_info.Element;
      //double version = build_info.Version;

      if (m_name == null)
        AttrError(node, "name");
      if (m_type == null)
        AttrError(node, "type");

      m_unit = Utils.XmlGetAttributeValue(node,"unit");

      if (m_type == "ustring")
        m_etype = SimpleOptionInfoType.TypeString;
      else if (m_type == "string") {
          m_etype = SimpleOptionInfoType.TypeString;
      }
      else if (m_type == "cstring") {
          throw new ArgumentException("Type 'cstring' for option is no more valid");
      }
      else if (m_type == "real")
        m_etype = SimpleOptionInfoType.TypeReal;
      else if (m_type == "bool")
        m_etype = SimpleOptionInfoType.TypeBool;
      else if (m_type == "integer")
        m_etype = SimpleOptionInfoType.TypeInteger;
      else if (m_type == "int32")
        m_etype = SimpleOptionInfoType.TypeInt32;
      else if (m_type == "int64")
        m_etype = SimpleOptionInfoType.TypeInt64;
      else if (m_type == "real2")
        m_etype = SimpleOptionInfoType.TypeReal2;
      else if (m_type == "real2x2")
        m_etype = SimpleOptionInfoType.TypeReal2x2;
      else if (m_type == "real3")
        m_etype = SimpleOptionInfoType.TypeReal3;
      else if (m_type == "real3x3")
        m_etype = SimpleOptionInfoType.TypeReal3x3;
      else if (m_type == "string[]")
        m_etype = SimpleOptionInfoType.TypeStringArray;
      else if (m_type == "real[]")
        m_etype = SimpleOptionInfoType.TypeRealArray;
      else if (m_type == "bool[]")
        m_etype = SimpleOptionInfoType.TypeBoolArray;
      else if (m_type == "integer[]")
        m_etype = SimpleOptionInfoType.TypeIntegerArray;
      else if (m_type == "int32[]")
        m_etype = SimpleOptionInfoType.TypeInt32Array;
      else if (m_type == "int64[]")
        m_etype = SimpleOptionInfoType.TypeInt64Array;
      else if (m_type == "real2[]")
        m_etype = SimpleOptionInfoType.TypeReal2Array;
      else if (m_type == "real2x2[]")
        m_etype = SimpleOptionInfoType.TypeReal2x2Array;
      else if (m_type == "real3[]")
        m_etype = SimpleOptionInfoType.TypeReal3Array;
      else if (m_type == "real3x3[]")
        m_etype = SimpleOptionInfoType.TypeReal3x3Array;
      else {
        Console.WriteLine("** ERREUR: Type de l'option <" + node.Name
             + "> inconnu (" + m_type + ").\n");
        Console.WriteLine("** Les types reconnus sont 'ustring', 'string', 'real', "
                          + "'bool', 'integer', 'int32', 'int64', 'real2', 'real3', 'real2x2', 'real3x3', "
                          + "'cstring', 'string[]', 'real[]',"
                          + "'real2[]', 'real3[]', 'real2x2[]', 'real3x3[]', "
                          + "'integer[]', 'int32[]', 'int64[]' .\n");
        Error(node, "mauvaise valeur pour l'attribut \"type\"");
      }
    }

    public override void Accept(IOptionInfoVisitor v)
    { v.VisitSimple(this); }

    SimpleOptionInfoType m_etype;
    //! Valeur de l'attribut XML "type", type de base de l'option simple.
    public SimpleOptionInfoType SimpleType { get { return m_etype; } }

    string m_unit;
    /// <summary>
    /// Valeur de l'attribut XML "unit" inidiquant le type de l'unité physique de la variable ou null si aucune.
    /// </summary>
    public string PhysicalUnit { get { return m_unit; } }
  }
}

