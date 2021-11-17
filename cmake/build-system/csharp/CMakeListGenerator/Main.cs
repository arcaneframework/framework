using System;
using System.IO;

namespace CMakeListGenerator
{ 
  public sealed class GlobalContext
  {
    private static readonly GlobalContext _instance = new GlobalContext ();
    
    private GlobalContext ()
    {
    }
    
    public static GlobalContext Instance {
      get {
        return _instance;
      }
    }
    
    private bool verbose = false;
    
    public bool Verbose {
      get { return verbose; }
      set { verbose = value; }
    }
  }

  #region EXCEPTION
  [Serializable]
  public class CodeException : Exception
  {
    public CodeException ()
    {
    }
    
    public CodeException (string message)
      : base(message)
    {
      ;
    }
  }
  
  [Serializable]
  public class CodeExit : Exception
  {
    public CodeExit ()
    {
    }
    
    public CodeExit (string message)
      : base(message)
    {
      ;
    }
  }
  #endregion

  class MainClass
  {
    private static int RTCODE_OK = 0;
    private static int RTCODE_ERROR = 1;

    #region CONFIGURATIONS
    static string codename = "CMakeListGenerator.exe";
    enum Action { None, RecursiveConvert, Convert, Generate };
    #endregion

    public static void Main (string[] args)
    {
      bool show_help = false;
      bool force_overwrite = false;
      Action action = Action.None;
     
      var options = new Mono.Options.OptionSet () {
          { "v|verbose", "Verbose mode", v => GlobalContext.Instance.Verbose = true },
          { "h|help",    "Show help page", v => show_help = true },
          { "convert",   "Convert config.xml to CMakeLists.txt", v => action = Action.Convert },
          { "recursive-convert",   "Convert recursively config.xml to CMakeLists.txt", v => action = Action.RecursiveConvert },
          { "generate",  "Generate CMakeLists.txt from current directory files", v => action = Action.Generate },
          { "f|force",   "Force overwrite if target file already exists", v => force_overwrite = true },
      };
        
      try {
        try {
          
          var extras = options.Parse (args);

          // Check if remaining args looks like options
          foreach (var arg in extras) {
            if (arg.StartsWith ("-")) {
              showHelpAndExit (options, String.Format("Invalid option {0} after --", arg));
            }
          }

          if (extras.Count == 0)
            showHelpAndExit (options, "Expected lib-path argument");

          String libname = extras[0];
          String path = ".";
          if (extras.Count > 1)
            path = extras[1];

          String outpath = path;
          if (extras.Count > 2)
            outpath = extras[2];

          if (extras.Count > 3)
            showHelpAndExit (options, "Unexpected argument");

          if (show_help)
            showHelpAndExit (options);

          switch (action) {
          case Action.RecursiveConvert:
            new Converter(force_overwrite, libname, path, outpath, true);
            break;
          case Action.Convert:
            new Converter(force_overwrite, libname, path, outpath, false);
            break;
          case Action.Generate:
            new Generator(force_overwrite, libname, path);
            break;
          default:
            showHelpAndExit (options, "you have to choose an action between --convert and --generate");
            break;
          }

          Environment.Exit (RTCODE_OK);
          
        } catch (Mono.Options.OptionException e) {
          showHelpAndExit (options, e.Message);
        }

      } catch (CodeExit e) {

        Console.WriteLine (e.Message);
        Environment.Exit (RTCODE_OK);

      } catch (Exception e) {

        Console.Error.WriteLine (String.Format ("Exiting with an error: {0}", e.Message));
        Environment.Exit (RTCODE_ERROR);

      }
    }

    private static void showHelpAndExit (Mono.Options.OptionSet options, String message = null)
    {
      StringWriter writer = new StringWriter ();
      if (message == null)
        writer.WriteLine ("Requested Help page");
      else
        writer.WriteLine (message);
      
      writer.WriteLine ("Usage: {0} [options] --generate|--convert --rootlib|--subdir [--] lib-name [path] [outpath]", codename);
      writer.WriteLine ("  lib-name defines library target name");
      writer.WriteLine ("  path defines working directory (default is .)");
      writer.WriteLine ("  outpath defines output directory (default is .)");
      writer.WriteLine ("Options : ");
      options.WriteOptionDescriptions (writer);

      if (message == null)
        throw new CodeExit (writer.ToString ());
      else
        throw new CodeException (writer.ToString ());
    }
  }
}
