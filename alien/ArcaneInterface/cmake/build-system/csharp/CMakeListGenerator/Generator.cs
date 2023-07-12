using System;
using System.IO;
using System.Collections.Generic;

namespace CMakeListGenerator
{
  public class InstructionFile
  {
    public InstructionFile(string RequestedFilename, bool RequestedHasHeader)
    {
      Filename = RequestedFilename;
      HasHeader = RequestedHasHeader;
    }
    public InstructionFile(string RequestedFilename)
    {
      Filename = RequestedFilename;
      HasHeader = true;
    }
    public string Filename { get; private set; }
    public bool HasHeader { get; private set; }
  }

  public class Generator
  {
    public Generator (bool force_overwrite, String libname, String path)
    {
      var infos = new DirectoryInfo(path);

      var Directories = new List<Directory>();
      var HeaderFiles = new List<string>();
      var AxlFiles = new List<string>();
      var CppFiles = new List<string>();
      var CFiles = new List<string>();
      var ModuleFiles = new List<string>();
      var ServiceFiles = new List<string>();
      
      foreach (var info in infos.GetDirectories()) {
        if (!info.Name.StartsWith (".")) {
          Directory dir = new Directory ();
          dir.Value = info.Name;
          Directories.Add (dir);
        }
      }

      List<string> AllCppFiles = new List<string>();
      foreach(var info in infos.GetFiles("*.cc")) 
        AllCppFiles.Add(info.Name.Replace(info.Extension,""));
      foreach(var info in infos.GetFiles("*.cpp")) 
        AllCppFiles.Add(info.Name.Replace(info.Extension,""));
      foreach(var info in infos.GetFiles("*.c")) 
        CFiles.Add(info.Name.Replace(info.Extension,""));
      foreach(var info in infos.GetFiles("*.h")) 
        HeaderFiles.Add(info.Name.Replace(info.Extension,""));
      foreach(var info in infos.GetFiles("*.axl")) 
        AxlFiles.Add(info.Name.Replace(info.Extension,""));
      
      foreach(var s in AllCppFiles)
      {
        if (EndsWithModule(s)) 
        {
          bool HasHeader = (HeaderFiles.Contains(s));
          string ModuleName = s.Replace("Module","");
          ModuleFiles.Add(ModuleName);
          if (HasHeader) HeaderFiles.Remove(s);
          AxlFiles.Remove(ModuleName);
        } 
        else if (EndsWithService(s))
        {
          bool HasHeader = (HeaderFiles.Contains(s));
          string ServiceName = s.Replace("Service","");
          ServiceFiles.Add(ServiceName);
          if (HasHeader) HeaderFiles.Remove(s);
          AxlFiles.Remove(ServiceName);
        } 
        else
        {
          bool HasHeader = (HeaderFiles.Contains(s));
          CppFiles.Add(s);
          if (HasHeader) HeaderFiles.Remove(s);
        }
      }

      CMakeListGenerator.CMakeList cmake = new CMakeListGenerator.CMakeList(new Makefile(), libname);
      cmake.Directories = Directories;
      cmake.AxlFiles = AxlFiles;
      cmake.CppFiles = CppFiles;
      cmake.CFiles = CFiles;
      cmake.ModuleFiles = ModuleFiles;
      cmake.ServiceFiles = ServiceFiles;

      String cmake_file = Path.Combine (path, "CMakeLists.txt");

      if (System.IO.File.Exists (cmake_file)) {
        if (force_overwrite)
          Console.WriteLine("WARNING: target '{0}' is going to be overwritten as requested by user", cmake_file);
        else
          throw new CodeException(String.Format("Cannot overwrite existing '{0}' file; use -f to force overwrite", cmake_file));          
      }

      using (StreamWriter writer = new StreamWriter(cmake_file)) {
        writer.Write (cmake.TransformText ());
      }
    }

    private static bool EndsWithModule(string s)
    {
      return ((s.Length > 5) && (s.Substring(s.Length - 6) == "Module"));
    }
    
    private static bool EndsWithService(string s)
    {
      return ((s.Length > 6) && (s.Substring(s.Length - 7) == "Service"));
    }
  }
}

