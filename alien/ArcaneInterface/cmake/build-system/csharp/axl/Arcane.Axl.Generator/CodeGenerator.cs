/*---------------------------------------------------------------------------*/
/* CodeGenerator.cc                                            (C) 2000-2007 */
/*                                                                           */
/* Classe de base des classes de génération de code.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Arcane.Axl
{
  /**
   * Classe de base des classes de génération de code. 
   */
  public abstract class CodeGenerator
  {
    /** Emplacement de la classe générée (pour les include). */
    protected string m_path;
    /** Repertoire de sortie des fichiers generes */
    protected string m_output_path;
    /** Nom de la classe générée. */
    protected string m_class_name;

    protected static string ARCANE_VERSION_STR = "0.8";

    protected CodeGenerator(string path,string output_path)
    {
      m_output_path = output_path;
      // Change les '_' par des '/'
      m_path = path.Replace("_", "/");
    }
    public abstract void writeFile();

    /**
     * Ecrit 2 lignes de commentaires horizontales dans le flux passé en
     * argument.
     * @param ts le flux dans lequel sera écrit le commentaire.
     */
    public static void WriteComments(TextWriter ts)
    {
      ts.Write("\n/*-------------------------------------");
      ts.Write("--------------------------------------*/\n");
      ts.Write("/*-------------------------------------");
      ts.Write("--------------------------------------*/\n\n");
    }
    /**
     * Ecrit des lignes de commentaires pour une en-tete de fichier
     * dans le flux passé en argument.
     * @param ts le flux dans lequel sera écrit le commentaire.
     */
    public static void WriteInfo(TextWriter ts)
    {
      string begin = "//";
      string current_date = DateTime.Now.ToString();
      ts.Write("/*-------------------------------------");
      ts.Write("--------------------------------------*/\n");
      ts.Write("/*-------------------------------------");
      ts.Write("--------------------------------------*/\n");
      ts.Write(begin + " #WARNING#: This file has been generated ");
      ts.Write("automatically. Do not edit.\n");
      ts.Write(begin + " Arcane version " + ARCANE_VERSION_STR);
      ts.Write(" : " + current_date + "\n");
      ts.Write("/*-------------------------------------");
      ts.Write("--------------------------------------*/\n");
      ts.Write("/*-------------------------------------");
      ts.Write("--------------------------------------*/\n\n");
    }

    /**
     * Transforme la chaine de caractéres "name" en un nom de fonction respectant
     * la typographie Arcane. Par exemple, "item-kind" devient "itemKind".
     * @param name le nom de la chaine d'origine.
     * @return la chaine transformée.
     */
    public static string ToFuncName(string name)
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

    /**
     * Transforme la chaine de caractéres "name" en un nom de classe respectant
     * la typographie Arcane. Par exemple, "item-kind" devient "ItemKind".
     * @param name le nom de la chaine d'origine.
     * @return la chaine transformée.
     */
    public static string ToClassName(string name)
    {
      string s = ToFuncName(name);
      char[] std_name = s.ToCharArray();
      if (std_name.Length > 0)
        std_name[0] = Char.ToUpper(std_name[0]);
      return new string(std_name);
    }
  }
}
