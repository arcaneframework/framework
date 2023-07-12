using System;
using System.Text;
using System.Linq;

namespace Arcane.Axl
{
  public static class StringExtensions
  {
    public static string ToLowerWithDash(this string value)
    {
      var sb = new StringBuilder();
      var s = value.ToCharArray();
      for (int i = 0; i < s.Length; ++i) {
        char ch = s[i];
        char lo_ch = Char.ToLower(s[i]);
        if (lo_ch!=ch && i!=0)
          sb.Append('-');
        sb.Append(lo_ch);
      }
      return sb.ToString();
    }

    public static string DashToUnderscore(this string input)
    {
      var sb = new StringBuilder();
      var s = input.ToCharArray();
      for (int i = 0; i < s.Length; ++i) {
        char ch = s[i];
        if (ch=='-') {
          sb.Append('_');
        }
        else {
          sb.Append(ch);
        }
      }
      return sb.ToString();
    }

    public static string FirstCharToUpper(this string input)
    {
      if (String.IsNullOrEmpty(input))
        throw new ArgumentException("Error in string extension method FirstCharToUpper !");
      return input.First().ToString().ToUpper() + input.Substring(1);
    }

    public static string ToFuncName(this string name)
    {
      char[] std_name = name.ToCharArray();
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < std_name.Length; ++i) {
        char ch = std_name[i];
        if (ch == '-' && (i + 1) < std_name.Length) {
          char next = Char.ToUpper(std_name[i + 1]);
          sb.Append(next);
          ++i;
        }
        else
          sb.Append(ch);
      }
      return sb.ToString();
    }

    public static string ToClassName(this string name)
    {
      string s = ToFuncName(name);
      char[] std_name = s.ToCharArray();
      if (std_name.Length > 0)
        std_name[0] = Char.ToUpper(std_name[0]);
      return new string(std_name);
    }

    // deplacer les constantes
    public static string ToArrayType(this string subType)
    {
      return SimpleTypeExtensions.Namespace()+"::"+"ConstArrayView<" + subType +">";
    }

    public static string ToUniqueArrayType(this string subType)
    {
      return SimpleTypeExtensions.Namespace()+"::"+"UniqueArray<" + subType +">";
    }

    static public string ConvertType(this string name)
    {
        return name.Replace(".", "::");
    }
  }
}

