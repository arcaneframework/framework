using System;
using System.Collections.Generic;
using System.IO;
using System.Xml;

namespace Arcane.Axl
{
  /// <summary>
  /// Classe generant les fichiers axl finaux qui seront
  /// utilises pour la documentation.
  /// Les traitements sont les suivants:
  /// - recupere les infos de la base de donnees Arcane
  /// - recupere l'ensemble des fichiers axl de depart.
  /// - genere dans un repertoire de sortie un fichier axl
  /// pour chaque implementation d'un service de type CaseOption.
  /// Eventuellement, si un service n'a pas de jeu de donnees,
  /// genere un axl vide.
  /// </summary>
  public class FinalAxlGenerator
  {
    CodeInfo m_code_info;
    string m_directory_name;
    string m_empty_xml = "<?xml version='1.0'?>\n<service name='Toto'>\n<name lang='fr'>Toto</name>\n<options/>\n</service>\n";
    public FinalAxlGenerator(CodeInfo code_info,string directory_name)
    {
      m_code_info = code_info;
      m_directory_name = directory_name;
    }

    public void Generate(string output_path)
    {
      Console.WriteLine("OUTPUT_PATH={0}",output_path);
      string final_path = output_path;
      Directory.CreateDirectory(final_path);

      Console.WriteLine("DIRECTORY={0}",m_directory_name);

      Console.WriteLine("SERVICES");
      int index = 0;
      Dictionary<CodeServiceInfo,string> new_service_axl_path = new Dictionary<CodeServiceInfo, string>();

      foreach(CodeServiceInfo csi in m_code_info.Services){
        new_service_axl_path.Add(csi,csi.FileBaseName);
      }

      // Genere les services
      foreach(CodeServiceInfo csi in m_code_info.Services){
        Console.WriteLine("SERVICE name={0} file={1}",csi.Name,csi.FileBaseName);
        string[] interfaces_name = csi.InterfacesName;
        bool is_invalid = false;
        XmlDocument doc = new XmlDocument();
        if (!String.IsNullOrEmpty(csi.FileBaseName)){
          is_invalid = _ReadAxl(doc,csi.FileBaseName+".axl");
        }
        else
          doc.LoadXml(m_empty_xml);
        if (is_invalid)
          continue;
        _SetAxlServiceName(doc,csi.Name);
        if (interfaces_name.Length==0){
          Console.WriteLine("WARNING: service '{0}' has no interface",csi.Name);
          interfaces_name = new string[] { csi.Name };
        }
        bool is_first = true;
        foreach(string iname in interfaces_name){
          Console.WriteLine("** INTERFACE name={0}",iname);
          _SetAxlInterfaceName(doc,iname);
          string fname = csi.FileBaseName;
          if (!is_first){
            // S'il y a plusieurs interfaces par service, il faut changer
            // le nom des fichiers pour ne pas ecraser celui d'avant
            fname = fname + "_" + index;
          }
          string full_name = Path.Combine(final_path,fname+".axl");
          ++index;
          Console.WriteLine("-- Saving XML document '{0}'",full_name);
          doc.Save(full_name);
          if (is_first){
            new_service_axl_path[csi] = fname;
            is_first = false;
          }
        }
      }

      // Genere les modules.
      // Pour eux, il ne faut rien changer. Il faut juste les copier
      // dans le repertoire de destination
      DirectoryInfo dir = new DirectoryInfo(m_directory_name);
      foreach(FileInfo fi in dir.GetFiles("*.axl")){
        string file_name = fi.Name;
        XmlDocument doc = new XmlDocument();
        if (_ReadAxl(doc,file_name))
          continue;
        XmlElement root_elem = doc.DocumentElement;
        if (root_elem==null)
          continue;
        if (root_elem.Name=="module"){
          Console.WriteLine("COPY FILE = '{0}'",file_name);
          File.Copy(fi.FullName,Path.Combine(final_path,file_name));
        }
      }

      // Enfin, regenere un fichier equivalent a arcane_internal.xml mais
      // en changeant les FileBaseName des services car ils ont change.
      using(TextWriter tw = File.CreateText(Path.Combine(final_path,"final_internal.xml"))){
        tw.WriteLine("<?xml version='1.0'?>");
        tw.WriteLine("<root>\n<services>");
        foreach(var v in new_service_axl_path){
          Console.WriteLine("CHANGE VALUE: {0} -> {1}",v.Key.FileBaseName,v.Value);
        }
        foreach(CodeServiceInfo csi in m_code_info.Services){
          tw.WriteLine("  <service name='{0}' file-base-name='{1}'>",csi.Name,new_service_axl_path[csi]);
          foreach(string s in csi.InterfacesName){
            tw.WriteLine("    <implement-class name='{0}' />",s);
          }
          tw.WriteLine("  </service>");
        }
        tw.WriteLine("<services-class>");
        foreach(CodeInterfaceInfo cii in m_code_info.Interfaces.Values){
          tw.WriteLine(" <class name='{0}'>",cii.Name);
          foreach(CodeServiceInfo csi in cii.Services)
            tw.WriteLine("    <service file-base-name='{0}' name='{1}'/>",new_service_axl_path[csi],csi.Name);
          tw.WriteLine("  </class>");
        }
        tw.WriteLine("</services-class>");

        tw.WriteLine("</services>\n</root>");
      }
    }

    bool _ReadAxl(XmlDocument doc,string name)
    {
      bool is_invalid = false;
      string axl_file_name = Path.Combine(m_directory_name,name);
      try{
        Console.WriteLine("Reading XML document file_name={0}",axl_file_name);
        doc.Load(axl_file_name);
      }
      catch(Exception ex){
        Console.WriteLine("Can not read xml file name={0} ex={1} stack={2}",axl_file_name,ex.Message,ex.StackTrace);
        is_invalid = true;
        doc = null;
      }
      return is_invalid;
    }

      /// <summary>
    /// Change dans le document \a doc le nom du service
    /// ainsi que sa traduction en le nom \a name
    /// </summary>
    /// <param name="doc">
    /// A <see cref="XmlDocument"/>
    /// </param>
    /// <param name="name">
    /// A <see cref="System.String"/>
    /// </param>
    void _SetAxlServiceName(XmlDocument doc,string name)
    {
      XmlElement root = doc.DocumentElement;
      if (root==null)
        return;
      root.SetAttribute("name",name);
      foreach(XmlNode node in root.ChildNodes){
        XmlElement name_elem = node as XmlElement;
        if (name_elem==null)
          continue;
        if (name_elem.Name!="name")
          continue;
        name_elem.InnerText = name;
      }
    }
    void _SetAxlInterfaceName(XmlDocument doc,string interface_name)
    {
      XmlElement root = doc.DocumentElement;
      if (root==null)
        return;
      XmlElement elem = Utils.GetElementIfExists(root,"interface");
      if (elem==null){
        elem = doc.CreateElement("interface");
        root.AppendChild(elem);
      }
      elem.SetAttribute("name",interface_name);
    }

  }
}

