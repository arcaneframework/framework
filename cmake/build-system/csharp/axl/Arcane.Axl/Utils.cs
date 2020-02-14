using System;
using System.Collections.Generic;
using System.Configuration;
using System.Text;
using System.Xml;
using System.IO;
using System.Reflection;

namespace Arcane.Axl
{
  static class Utils
  {
    public static bool XmlParseStringToBool(string s)
    {
      if (s == "0" || s == "false")
        return false;
      if (s == "1" || s == "true")
        return true;
      throw new FormatException("Can not convert '" + s + "' to boolean");
    }
    public static bool XmlParseNode(XmlNode node, bool default_value)
    {
      if (node == null)
        return default_value;
      return XmlParseStringToBool(node.Value);
    }

    public static string XmlGetAttributeValue(XmlElement element,string name)
    {
      if (element==null)
        return null;
      XmlNode attr = element.GetAttributeNode(name);
      if (attr==null)
        return null;
      return attr.Value;
    }
    /**
     * \brief Récupère un élément s'il existe.
     *
     * Récupère un élément de nom \a elem_name fils de \a parent.
     * Si l'élément n'existe pas, null est retourné.
     */
    public static XmlElement GetElementIfExists(XmlElement parent, string elem_name)
    {
      if (parent == null)
        throw new ArgumentException("Bad value for argument 'parent'");
      foreach (XmlNode i in parent) {
        if (i.NodeType != XmlNodeType.Element)
          continue;
        if (i.LocalName == elem_name)
          return i as XmlElement;
      }
      return null;
    }

    /**
     * \brief Récupère les valeurs d'une liste d'éléments
     *
     * Récupère les valeurs des éléments de nom \a elem_name fils de \a parent.
     */
    public static string[] GetElementsValue(XmlElement parent,string elem_name)
    {
      if (parent == null)
        throw new ArgumentException("Bad value for argument 'parent'");
      List<string> values = new List<string>();
      foreach (XmlNode i in parent) {
        if (i.NodeType != XmlNodeType.Element)
          continue;
        if (i.LocalName == elem_name)
          values.Add(i.InnerText);
      }
      return values.ToArray();
    }

    public static string ToLowerWithDash(string upper_name)
    {
      StringBuilder sb = new StringBuilder();
      char[] s = upper_name.ToCharArray();
      for( int i=0; i<s.Length; ++i ){
        char ch = s[i];
        char lo_ch = Char.ToLower(s[i]);
        if (lo_ch!=ch && i!=0)
          sb.Append('-');
        sb.Append(lo_ch);
      }
      return sb.ToString();
    }

    public static Stream GetAxlSchemaAsStream()
    {
      Stream stream = Assembly.GetAssembly(typeof(Utils)).GetManifestResourceStream("axl.xsd");
      if (stream == null)
        throw new ApplicationException("Can not find embedded schema file 'axl.xsd'");
      return stream;
    }
    static Encoding m_write_encoding = Encoding.Default;
    public static Encoding WriteEncoding { get{ return m_write_encoding; } set { m_write_encoding = value; } }
  }
}
