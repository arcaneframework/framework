using System;
using System.Linq;
using System.Collections.Generic;

namespace CMakeListGenerator
{
  public partial class CMakeList
  {
    public CMakeListGenerator.Makefile Makefile { get; private set; }
    
    public String LibraryName { get; private set; }
    public String SrcPath { get; set; }
    public String OutSrcPath { get; set; }

    public CMakeList (CMakeListGenerator.Makefile makefile,
                      String library_name)
    {
      Makefile = makefile;
      LibraryName = library_name;
      SrcPath = "";
    }

    public IEnumerable<Package> NeededPackages {
      get {
        if(Makefile.neededpackages == null) {
          return new List<Package> ();
        }
        return Makefile.neededpackages;
      }
    }

    public IEnumerable<Directory> Directories {
      get {
        if(Makefile.directories == null) {
          return new List<Directory> ();
        }
        return Makefile.directories;
      }
      set {
        var newDirs = new List<Directory>(Makefile.directories) ?? new List<Directory>();
        newDirs.AddRange (value);
        Makefile.directories = newDirs.ToArray ();
      }
    }

    public IEnumerable<String> LawServiceFiles {
      get {
        return Makefile.files.Files (default(Language).ServiceLaw ());
      }
      set {
        extendFiles(value, default(Language).ServiceLaw ());
      }
    }
   
    public IEnumerable<String> LawServiceHeaders {
      get {
        return Makefile.files.Headers (default(Language).ServiceLaw ());
      }
    }

    public IEnumerable<String> ModuleFiles {
      get {
        return Makefile.files.Files (default(Language).Module ());
      }
      set {
        extendFiles(value, default(Language).Module ());
      }
    }

    public IEnumerable<String> ModuleHeaders {
      get {
        return Makefile.files.Headers (default(Language).Module ());
      }
    }

    public IEnumerable<String> ServiceFiles {
      get {
        return Makefile.files.Files (default(Language).Service ());
      }
      set {
        extendFiles(value, default(Language).Service ());
      }
    }

    public IEnumerable<String> ServiceHeaders {
      get {
        var headers = Makefile.files.Headers (default(Language).Service ());
        foreach(var h in headers) {
          Console.WriteLine("service header = {0}", h);
        }
        return Makefile.files.Headers (default(Language).Service ());
      }
    }

    public IEnumerable<String> CppFiles {
      get {
          return Makefile.files.Files (default(Language).Cpp ());
      }
      set {
        extendFiles(value, default(Language).Cpp ());
      }
    }

    public IEnumerable<String> CppHeaders {
      get {
        var headers = new List<String> ();
        headers.AddRange(Makefile.files.Files (default(Language).CppHeader ()));
        headers.AddRange(Makefile.files.Headers (default(Language).Cpp ()));
        return headers;
      }
    }

    public IEnumerable<String> CFiles {
      get {
        return Makefile.files.Files (default(Language).C ());
      }
      set {
        extendFiles(value, default(Language).C ());
      }
    }

    public IEnumerable<String> CHeaders {
      get {
        if(Makefile.files == null) {
          return new List<String> ();
        }
        var headers = new List<String> ();
        headers.AddRange(Makefile.files.Files (default(Language).CHeader ()));
        headers.AddRange(Makefile.files.Headers (default(Language).C ()));
        return headers;
      }
    }

    public IEnumerable<String> CSharpFiles {
      get {
        return Makefile.files.Files (default(Language).CSharp ());
      }
      set {
        extendFiles(value, default(Language).CSharp ());
      }
    }

    public IEnumerable<String> AxlFiles {
      get {
        var axls = new List<string> ();
        axls.AddRange (Makefile.files.Files (default(Language).Axl ()));
        axls.AddRange (ModuleFiles);
        axls.AddRange (ServiceFiles);
        return axls;
      }
      set {
        extendFiles(value, default(Language).Axl ());
      }
    }

    public String Depend {
      get {
        return Makefile.depend;
      }
    }

    private void extendFiles(IEnumerable<String> values, Language language) {
      List<File> value_files = new List<File>();
      foreach(var f in values) {
        CMakeListGenerator.File file = new CMakeListGenerator.File();
        file.Value = f;
        value_files.Add (file);
      }
      Files files = new Files();
      files.language = language;
      files.file = value_files.ToArray();
      var newFiles = new List<Files> ();
      if (Makefile.files != null)
        newFiles = new List<Files>(Makefile.files);
      newFiles.Add (files);
      Makefile.files = newFiles.ToArray ();
    }

    private static String ConvertDependencies(String value) {
      String[] dependencies = value.Split (new char[] { ' ', ',' }, StringSplitOptions.RemoveEmptyEntries);
      String cmake_condition = null;
      for(int i = 0; i < dependencies.Length; ++i) {
        String true_dep = dependencies[i];
        bool test_dep = true;
        while (true_dep[0] == '!') {
          test_dep = !test_dep;
          true_dep = true_dep.Substring (1);
        }
        if (cmake_condition == null)
          cmake_condition = ((test_dep)?"":"NOT ") + "TARGET " + true_dep;
        else
          cmake_condition += " AND " + ((test_dep)?"":"NOT ") + "TARGET " + true_dep;
      }
      return cmake_condition;
    }
  }
}
