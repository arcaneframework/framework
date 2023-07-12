/*---------------------------------------------------------------------------*/
/* DoxygenExampleVisitor.cs                              (C) 2000-2007 */
/*                                                                           */
/* Génération de la documentation au format Doxygen.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace Arcane.Axl
{
  public class DoxygenExampleVisitor
   : IOptionInfoVisitor
  {
    private TextWriter m_example_stream;
    private string m_lang;
    private int m_level;
    private bool m_do_aliases;
    private CodeInfo m_code_info;
    DoxygenExampleFile m_doc_file;

    public DoxygenExampleVisitor(DoxygenExampleFile doc_file, CodeInfo code_info, string lang)
    {
      _DoxygenExampleVisitor(doc_file, code_info, lang, 0, true);
    }

    public DoxygenExampleVisitor(DoxygenExampleFile doc_file, CodeInfo code_info, string lang, int level, bool do_aliases)
    {
      _DoxygenExampleVisitor(doc_file, code_info, lang, level, do_aliases);
    }

    private void _DoxygenExampleVisitor(DoxygenExampleFile doc_file, CodeInfo code_info, string lang, int level, bool do_aliases)
    {
      m_doc_file = doc_file;
      m_example_stream = doc_file.ExampleStream;
      m_code_info = code_info;
      m_lang = lang;
      m_level = level;
      m_do_aliases = do_aliases;
    }
    
    public void VisitServiceOrModule(ServiceOrModuleInfo info)
    {
      // Ecriture dans le fichier global au service ou au module
      if (info.IsModule)
        m_example_stream.WriteLine("<{0}>", info.GetTranslatedName(m_lang));
      foreach (Option option in info.Options) {
        option.Accept(this);
      }
      if (info.IsModule) {
        m_example_stream.WriteLine("</{0}>", info.GetTranslatedName(m_lang));
        // Ecriture des balises de début et de fin dans un fichier dédié
        DoxygenExampleFile df = new DoxygenExampleFile(info, "_intro");
        df.ExampleStream.Write("<{0}>\n  <!-- Liste des options -->\n</{0}>\n", info.GetTranslatedName(m_lang));
        df.Write();
      }
    }

    public void VisitComplex(ComplexOptionInfo option)
    {
      m_level += 1;
      string shift = String.Empty;
      for (int iLevel = 0 ; iLevel < m_level ; ++iLevel) shift += "  ";
      string strBegOption = String.Format("{0}<{1}>", shift, option.GetTranslatedName(m_lang));
      string strOccurs = _getStringOccurs(option.MinOccurs, option.MaxOccurs);
      if (!String.IsNullOrEmpty(strOccurs))
        strBegOption += String.Format("  <!-- {0}->\n", strOccurs);
      else
        strBegOption += "\n";
      string strEndOption = String.Format("{0}</{1}>\n", shift, option.GetTranslatedName(m_lang));
      // Ecriture dans le fichier global au module
      m_example_stream.Write(strBegOption);
      option.AcceptChildren(this);
      m_example_stream.Write(strEndOption);
      if (m_level <= 1) {
        // Ecriture dans un fichier dédié à l'option
        DoxygenExampleFile df = new DoxygenExampleFile(option);
        DoxygenExampleVisitor dv = new DoxygenExampleVisitor(df, m_code_info, m_lang, m_level, false);
        df.ExampleStream.Write(strBegOption);
        option.AcceptChildren(dv);
        df.ExampleStream.Write(strEndOption);
        df.Write();
      }
      m_level -= 1;
      // Ecriture dans le fichier des alias
      _WriteAliasOption(option);
    }

    public void VisitEnumeration(EnumerationOptionInfo option)
    {
      string defaultValue = option.GetTranslatedEnumerationName(option.DefaultValue, m_lang);
      if (String.IsNullOrEmpty(defaultValue)) {
        Console.WriteLine("WARNING: enumeration option has not a default value: '{0}/{1}'",
                          option.ServiceOrModule.Name, option.FullName);
        defaultValue = "cf. choix";
        //defaultValue = option.EnumValues[0].GetTranslatedName(m_lang);
      }
      string stringEnumValues = string.Empty;
      foreach (EnumValueOptionInfo ev in option.EnumValues) {
        string ev_tn = ev.GetTranslatedName(m_lang);
        stringEnumValues += " " + ev_tn;
        // Ecriture dans le fichier des alias
        string alias = _getAliasPrefixe(option)  + "_" + ev_tn.Replace('-', '_');
        _WriteAlias(alias, string.Empty, ev_tn);
      }
      string stringOption = _getStringOption(option, defaultValue, "Choix:" + stringEnumValues);
      _WriteCommon(option, stringOption);
    }
    
    public void VisitExtended(ExtendedOptionInfo option)
    {
      string defaultValue = option.DefaultValue;
      if (String.IsNullOrEmpty(defaultValue))
        defaultValue = "NOM";
      string valueType;
      switch (option.Type) {
      case "CellGroup":
        valueType = "groupe de mailles";
        break;
      case "FaceGroup":
        valueType = "groupe de faces";
        break;
      case "NodeGroup":
        valueType = "groupe de noeuds";
        break;
      case "ItemGroup":
        valueType = "groupe d'entités de maillage";
        break;
      default:
        //valueType = option.Type;
        valueType = String.Empty;
        break;
      }
      if (!String.IsNullOrEmpty(valueType))
        valueType = valueType.Insert(0, "Type: ");
      string stringOption = _getStringOption(option, defaultValue, valueType);
      _WriteCommon(option, stringOption);
    }
 
    public void VisitSimple(SimpleOptionInfo option)
    {
      string defaultValue = option.DefaultValue;
      string valueType = string.Empty;
      switch (option.Type) {
      case "real":
        //valueType = "réel";
        if (String.IsNullOrEmpty(defaultValue))
          defaultValue = "réel";
        break;
      case "integer":
        //valueType = "entier";
        if (String.IsNullOrEmpty(defaultValue))
          defaultValue = "entier";
        break;
      case "bool":
        //valueType = "booléen";
        if (String.IsNullOrEmpty(defaultValue))
          defaultValue = "booléen";
        break;
      case "string":
      case "cstring":
      case "ustring":
        //valueType = "chaîne";
        if (String.IsNullOrEmpty(defaultValue))
          defaultValue = "NOM";
        break;
      default:
        //valueType = o.Type;
        break;
      }
      if (!String.IsNullOrEmpty(valueType))
        valueType = valueType.Insert(0, "Type: ");
      string stringOption = _getStringOption(option, defaultValue, valueType);
      _WriteCommon(option, stringOption);
    }

    public void VisitScript(ScriptOptionInfo option)
    {
      string stringOption = _getStringOption(option, string.Empty, string.Empty);
      _WriteCommon(option, stringOption);      
    }

    public void VisitServiceInstance(ServiceInstanceOptionInfo option)
    {
      string stringValues = string.Empty;
      string file_base_name_service = string.Empty;
      CodeInterfaceInfo interface_info = null;
      if (m_code_info.Interfaces.TryGetValue(option.Type, out interface_info)) {
        bool found = false;
        foreach (CodeServiceInfo csi in interface_info.Services) {
          if (option.DefaultValue == csi.Name) {
            file_base_name_service = csi.FileBaseName;
            found = true;
          }
          if (!String.IsNullOrEmpty(csi.Name)) {
            stringValues += " " + csi.Name;
            // Ecriture dans le fichier des alias
            string alias = _getAliasPrefixe(option)  + "_" + csi.Name.Replace('-', '_');
            _WriteAlias(alias, string.Empty, csi.Name);
          }
        }
        if (!found) {
            Console.WriteLine("WARNING: service default value not found '{0}' while treating option '{1}/{2}'",
                              option.DefaultValue, option.ServiceOrModule.Name, option.FullName);
        }
      }
      else {
        bool found = false;
        foreach (CodeServiceInfo si in m_code_info.Services) {
          if (si.Name == option.DefaultValue) {
            file_base_name_service = si.FileBaseName;
            found = true;
            break;
          }
        }
        if (! found) {
          Console.WriteLine("WARNING: service type not found '{0}' while treating option '{1}/{2}'",
                            option.Type, option.ServiceOrModule.Name, option.FullName);
        }
      }
      if (!String.IsNullOrEmpty(stringValues))
        stringValues = "Choix:" + stringValues;
      string serviceContent = string.Empty;
      if (!String.IsNullOrEmpty(file_base_name_service)) {
        string base_name = "axldoc_service_" + file_base_name_service.Replace('/', '_');
        string file_path = DoxygenExampleFile.getFilePath(base_name);
        try {
          Encoding encoding = Encoding.GetEncoding("iso-8859-1");
          TextReader tr = new StreamReader(file_path, encoding);
          serviceContent = tr.ReadToEnd();
          tr.Close();
        } catch (System.IO.FileNotFoundException) {
          Console.WriteLine("WARNING: file {0} not found while treating {1}", file_path, m_doc_file.BaseName);
        }
      }
      else {
        Console.WriteLine("WARNING: service base file name not found for service type '{0}' while treating option '{1}/{2}'",
                          option.Type, option.ServiceOrModule.Name, option.FullName);
      }
      string stringService = _getStringService(option, option.DefaultValue, stringValues, serviceContent);
      _WriteCommon(option, stringService);
    }

    private string _getStringService(Option option, string defaultValue, string type, string serviceContent)
    {
      string shift = "  ";
      for (int iLevel = 0 ; iLevel < m_level ; ++iLevel) shift += "  ";
      string comm = string.Empty;
      if (!String.IsNullOrEmpty(type))
        comm = String.Format("  <!-- {0} -->", type);
      string formattedContent;
      if (!String.IsNullOrEmpty(serviceContent)) {
        formattedContent = string.Empty;
        string[] lines = serviceContent.Split('\n');
        foreach (string line in lines) {
          string newLine = String.Format("{0}{1}\n", shift, line);
          formattedContent += newLine;
        }
        formattedContent = formattedContent.Trim('\n');
      }
      else
        formattedContent = shift;
      return String.Format("{0}<{1} name=\"{2}\">{3}\n{4}</{1}>\n",
                           shift,
                           option.GetTranslatedName(m_lang),
                           defaultValue,
                           comm,
                           formattedContent);
    }
   
    private string _getStringOption(Option option, string defaultValue, string type)
    {
      string shift = "  ";
      for (int iLevel = 0 ; iLevel < m_level ; ++iLevel) shift += "  ";
      string occurs = _getStringOccurs(option.MinOccurs, option.MaxOccurs);
      string comm = string.Empty;
      if (!String.IsNullOrEmpty(occurs) || !String.IsNullOrEmpty(type))
        comm = String.Format("  <!-- {0}{1} -->", occurs, type);
      return String.Format("{0}<{1}>{2}</{1}>{3}\n",
                           shift,
                           option.GetTranslatedName(m_lang),
                           defaultValue,
                           comm);
    }
    
    private string _getStringOccurs(int minOccurs, int maxOccurs)
    {
      string stringOccurs = string.Empty;
      if (maxOccurs > 1) {
        if (maxOccurs != minOccurs)
          stringOccurs = String.Format("entre {0} et {1} fois", minOccurs, maxOccurs);
        else
          stringOccurs = String.Format("{0} fois", maxOccurs);
      }
      else if (maxOccurs == 1 && minOccurs == 0) {
        stringOccurs = "0 ou 1 fois";
      }
      else if (maxOccurs == Option.UNBOUNDED) {
        if (minOccurs == 0)
          stringOccurs = "0, 1 ou plusieurs fois";
        else if (minOccurs == 1)
          stringOccurs = "1 ou plusieurs fois";
        else
          stringOccurs = String.Format("au moins {0} fois", minOccurs);
      }
      if (!String.IsNullOrEmpty(stringOccurs))
        stringOccurs = String.Format("Occurence: {0} - ", stringOccurs);
      return stringOccurs;
    }

    private void _WriteCommon(Option option, string stringOption)
    {
      m_level +=1;
      // Ecriture dans le fichier global au module
      m_example_stream.Write(stringOption);
      if (m_level <= 1) {
        // Ecriture dans un fichier dédié à l'option
        DoxygenExampleFile df = new DoxygenExampleFile(option);
        df.ExampleStream.Write(stringOption);
        df.Write();
      }
      m_level -= 1;
      // Ecriture dans le fichier des alias
      _WriteAliasOption(option);
    }

    private string _getAliasPrefixe(Option option)
    {
      return option.ServiceOrModule.GetTranslatedName(m_lang).Replace('-', '_') + "_"
        + option.GetTranslatedFullName(m_lang).Replace('/', '_').Replace('-', '_');
    }

    private void _WriteAliasOption(Option option)
    {
      _WriteAlias(_getAliasPrefixe(option),
                  DoxygenDocumentationUtils.AnchorName(option),
                  string.Format("\\<{0}\\>", option.GetTranslatedName(m_lang)));
    }

    private void _WriteAlias(string alias, string anchor, string content)
    {
      if (!String.IsNullOrEmpty(anchor)) {
        content = string.Format("\\ref {0} \\\"{1}\\\"", anchor, content);
      }
      if (m_do_aliases) {
        // Ecriture dans le fichier des alias
        DoxygenExampleFile.AliasesStream.WriteLine("ALIASES += {0}=\"<code class='OptionName'>{1}</code>\"",
                                                   alias, content);
      }
    }
  }
}
