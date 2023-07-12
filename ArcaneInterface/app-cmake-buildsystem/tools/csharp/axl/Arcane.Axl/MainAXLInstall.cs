using System;
using System.Collections.Generic;
using System.IO;
using System.Xml;

namespace Arcane.Axl
{
  /*!
   * \brief Installe les fichiers axl dans un répertoire cible.
   * 
   * Installe les fichiers en les préfixant de leur chemin relatif.
   * Cela permet ensuite de les utiliser pour générer la documentation.
   */
  public class MainAXLInstall
  {
    string m_output_path;
    string m_base_path;
    string[] m_sub_dirs;
    bool m_is_recursive;

    public MainAXLInstall(string base_path,string output_path,bool is_recursive,string[] sub_dirs)
    {
      m_base_path = base_path;
      m_sub_dirs = sub_dirs;
      m_output_path = output_path;
      m_is_recursive = is_recursive;
    }

    public static MainAXLInstall Create(string[] args)
    {
      int n = args.Length;
      string base_path = null;
      string output_path = null;
      bool is_recursive = false;
      List<string> sub_dirs = new List<string>();
      for( int i=0; i<n; ++i ){
        string s = args[i];
        if (s=="--help" || s=="-help") {
          _PrintUsage();
          return null;
        }
        if (s=="-b"){
          base_path = _GetArg(args,++i,"-b");
        }
        else if (s=="-o"){
          output_path = _GetArg(args,++i,"-o");
        }
        else if (s=="-r"){
          is_recursive = true;
        }
        else
          sub_dirs.Add(s);
      }
      if (String.IsNullOrEmpty(base_path))
        throw new ArgumentException("No base path specified");
      if (String.IsNullOrEmpty(output_path))
        throw new ArgumentException("No output path specified");

      MainAXLInstall rc = new MainAXLInstall(base_path,output_path,is_recursive,sub_dirs.ToArray());
      return rc;
    }

    public static string _GetArg(string[] args,int i,string opt_name)
    {
      if (i>=args.Length)
        throw new ArgumentException(String.Format("Bad argument for option '{0}'",opt_name));
      Console.WriteLine("GET_ARG opt='{0}' s='{1}'",opt_name,args[i]);
      return args[i];
    }

    public void Execute()
    {
      Console.WriteLine("BASE_PATH='{0}'",m_base_path);
      foreach(string s in m_sub_dirs){
        _Execute(s);
      }
    }

    void _Execute(string rel_path)
    {
      Console.WriteLine("REL_PATH='{0}'",rel_path);
      string full_path = Path.Combine(m_base_path,rel_path);
      DirectoryInfo dir = new DirectoryInfo(full_path);
      foreach(FileInfo fi in dir.GetFiles("*.axl")){
        Console.WriteLine("FI='{0}'",fi.Name);
        string fi_base_name = Path.GetFileNameWithoutExtension(fi.Name);
        string out_name = fi_base_name + "_" + rel_path.Replace(Path.DirectorySeparatorChar,'_') + ".axl";
        Console.WriteLine("REL_NAME='{0}'",out_name);
        string out_path = Path.Combine(m_output_path,out_name);
        _Copy(fi.FullName,out_path);
      }
      if (m_is_recursive){
        foreach(DirectoryInfo di in dir.GetDirectories()){
          _Execute(Path.Combine(rel_path,di.Name));
        }
      }
    }

    void _Copy(string input_path,string output_path)
    {
      try{
        Console.WriteLine("Copy '{0}' to '{1}'",input_path,output_path);
        File.Copy(input_path,output_path,true);
      }
      catch(Exception ex){
        Console.WriteLine("Can not copy file ex={0} stack={1}",ex.Message,ex.StackTrace);
      }
    }

    static void _PrintUsage()
    {
      Console.WriteLine("Usage: axlcopy install base_path output_path sub_dir1 sub_dir2 ...");
    }

  }
}
