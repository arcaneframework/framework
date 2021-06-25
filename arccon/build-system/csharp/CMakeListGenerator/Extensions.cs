using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

public static class StringExtension
{
    public static string CamelCaseName(this string name)
    {
        char[] std_name = name.ToCharArray();
        bool next_is_upper = true;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < std_name.Length; ++i)
        {
            char ch = std_name[i];
            if (ch == '-')
            {
                next_is_upper = true;
            }
            else if (next_is_upper)
            {
                sb.Append(Char.ToUpper(ch));
                next_is_upper = false;
            }
            else
            {
                sb.Append(ch);
            }
        }
        return sb.ToString();
    }
}

static class LanguageExtensions {

  public static CMakeListGenerator.Language Axl(this CMakeListGenerator.Language l) {
    return CMakeListGenerator.Language.axl;
  }

  public static CMakeListGenerator.Language Cpp(this CMakeListGenerator.Language l) {
    return CMakeListGenerator.Language.c;
  }
  
  public static CMakeListGenerator.Language CppHeader(this CMakeListGenerator.Language l) {
    return CMakeListGenerator.Language.cheader;
  }
  
  public static CMakeListGenerator.Language C(this CMakeListGenerator.Language l) {
    return CMakeListGenerator.Language.c1;
  }
  
  public static CMakeListGenerator.Language CHeader(this CMakeListGenerator.Language l) {
    return CMakeListGenerator.Language.cheader1;
  }
  
  public static CMakeListGenerator.Language CSharp(this CMakeListGenerator.Language l) {
    return CMakeListGenerator.Language.c2;
  }
  
  public static CMakeListGenerator.Language Module(this CMakeListGenerator.Language l) {
    return CMakeListGenerator.Language.module;
  }
  
  public static CMakeListGenerator.Language Service(this CMakeListGenerator.Language l) {
    return CMakeListGenerator.Language.service;
  }

  public static CMakeListGenerator.Language ServiceLaw(this CMakeListGenerator.Language l) {
    return CMakeListGenerator.Language.servicelaw;
  }

  public static CMakeListGenerator.Language F90(this CMakeListGenerator.Language l) {
    return CMakeListGenerator.Language.f90;
  }

  public static bool isAxl(this CMakeListGenerator.Language l) {
    return l == CMakeListGenerator.Language.axl;
  }

  public static bool isCpp(this CMakeListGenerator.Language l) {
    return l == CMakeListGenerator.Language.c;
  }

  public static bool isCppHeader(this CMakeListGenerator.Language l) {
    return l == CMakeListGenerator.Language.cheader;
  }

  public static bool isC(this CMakeListGenerator.Language l) {
    return l == CMakeListGenerator.Language.c1;
  }

  public static bool isCHeader(this CMakeListGenerator.Language l) {
    return l == CMakeListGenerator.Language.cheader1;
  }

  public static bool isCSharp(this CMakeListGenerator.Language l) {
    return l == CMakeListGenerator.Language.c2;
  }

  public static bool isModule(this CMakeListGenerator.Language l) {
    return l == CMakeListGenerator.Language.module;
  }
  
  public static bool isService(this CMakeListGenerator.Language l) {
    return l == CMakeListGenerator.Language.service;
  }

  public static bool isLawService(this CMakeListGenerator.Language l) {
    return l == CMakeListGenerator.Language.servicelaw;
  }

  public static bool isF90(this CMakeListGenerator.Language l) {
    return l == CMakeListGenerator.Language.f90;
  }
}

static class FilesExtensions {
  
  public static IEnumerable<String> Files (this CMakeListGenerator.Files[] files, CMakeListGenerator.Language lang) 
  {
    if(files == null) {
      return new List<String> ();
    }
    return files.Where(f => f.file != null && f.language == lang)
                .SelectMany (f => f.file)
                .Where (f => f.Value != null)
                .Select (f => f.Value);
  }

  public static IEnumerable<String> Headers (this CMakeListGenerator.Files[] files, CMakeListGenerator.Language lang) 
  {
    if(files == null) {
      return new List<String> ();
    }
    return files.Where(f => f.file != null && f.language == lang)
                .SelectMany (f => f.file)
                .Where (f => f.Value != null)
                .Where (f => f.headerSpecified == true )
                .Where (f => f.header == true)
                .Select (f => f.Value);
  }
}
