using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Text.RegularExpressions;
using System.Diagnostics;

namespace WindowsPathResolver
{
    class Program
    {
        private static int RTCODE_OK = 0;
        private static int RTCODE_ERROR = 1;

        static void Main(string[] args)
        {
            try
            {
                WindowsPathResolver c = new WindowsPathResolver();
                c.analyze(args);
                Environment.Exit(RTCODE_OK);
            }
            catch (WinPathResolverExit e)
            {
                Console.WriteLine(e.Message);
                Environment.Exit(RTCODE_OK);
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(String.Format("WindowsPathResolver exiting with an error: {0}", e.Message));
                Environment.Exit(RTCODE_ERROR);
            }
        }
    }

    #region EXCEPTION
    [Serializable]
    public class WinPathResolverException : Exception
    {
        public WinPathResolverException()
        {
        }

        public WinPathResolverException(string message)
            : base(message)
        {
            ;
        }
    }

    [Serializable]
    public class WinPathResolverExit : Exception
    {
        public WinPathResolverExit()
        {
        }

        public WinPathResolverExit(string message)
            : base(message)
        {
            ;
        }
    }
    #endregion

    public class WindowsPathResolver
    {
        #region CONFIGURATIONS
        string codename = "WindowsPathResolver.exe";
        #endregion

        #region MEMBERS
        enum Resolution { Path, Directory, File };
        enum PathStyle { CMake, Native }
        private Mono.Options.OptionSet m_options = null;
        bool m_show_help = false;
        bool m_verbose = false;
        Resolution m_resolution = Resolution.Path;
        PathStyle m_path_style = PathStyle.Native;
        #endregion
        
        public WindowsPathResolver()
        {
            m_options = new Mono.Options.OptionSet();
            m_options.Add("v|verbose", "Verbose mode", v => { m_verbose = true; });
            m_options.Add("f|file", "Want file resolution", v => { m_resolution = Resolution.File; });
            m_options.Add("d|directory", "Want directory resolution", v => { m_resolution = Resolution.Directory; });
            m_options.Add("cmake", "CMake style path", v => { m_path_style = PathStyle.CMake; });
            m_options.Add("h|help", "Show help page", v => { m_show_help = true; });
        }

        public void analyze(string[] args)
        {
            try
            {
                List<String> remaining_args = m_options.Parse(args);

                // Before directory/file checking
                if (m_show_help || remaining_args.Count != 1)
                    ShowHelpAndExit();

                string path = remaining_args.ToArray()[0];

                string resolved_path = resolvePath(path);
                if (resolved_path == null && !path.EndsWith(".lnk"))
                    resolved_path = resolvePath(path + ".lnk");

                if (resolved_path == null)
                {
                    throw new WinPathResolverException(String.Format("Cannot resolve path '{0}' as a {1}", path, m_resolution));
                }
                else
                {
                    if (m_path_style == PathStyle.CMake) 
                    {
                        if (m_verbose)
                            Console.Error.WriteLine("Converting path to CMake style...");
                        resolved_path = resolved_path.Replace("\\", "/");
                    }
                    Console.WriteLine(resolved_path);
                }
            }
            catch (Mono.Options.OptionException e)
            {
                ShowHelpAndExit(e.Message);
            }
        }

        private string resolvePath(String path)
        {
            if (Directory.Exists(path))
            {
                string resolved_path = Path.GetFullPath(path);
                if (m_verbose)
                    Console.Error.WriteLine("Directory '{0}' found", resolved_path);
                if (m_resolution == Resolution.File)
                    return null;
                else
                    return resolved_path;
            }

            try 
            {
                IWshRuntimeLibrary.IWshShell shell = new IWshRuntimeLibrary.WshShell();
                IWshRuntimeLibrary.IWshShortcut shortcut = (IWshRuntimeLibrary.IWshShortcut)shell.CreateShortcut(path);
                string resolved_path = shortcut.TargetPath;
                if (!String.IsNullOrEmpty(resolved_path))
                {
                    if (m_verbose)
                        Console.Error.WriteLine("LNK Shortcut '{0}' found from '{1}'", resolved_path, path);
                    return resolvePath(resolved_path);
                }
                else
                {
                    return null;
                }
            }
            catch (Exception e)
            {
                // if (m_verbose)
                //  Console.Error.WriteLine("Catch any exception while converting {0}", path);
            }
            
            if (File.Exists(path))
            {
                string resolved_path = Path.GetFullPath(path);
                if (m_verbose)
                    Console.Error.WriteLine("Normal file '{0}' found", resolved_path);
                if (m_resolution == Resolution.Directory)
                    return null;
                else
                    return resolved_path;
            }

            return null;
        }

        private void ShowHelpAndExit(String message = null)
        {
            StringWriter writer = new StringWriter();
            if (message == null)
                writer.WriteLine("Requested Help page");
            else
                writer.WriteLine(message);

            writer.WriteLine("Usage: {0} [options] Path", codename);
            writer.WriteLine("Options : ");
            m_options.WriteOptionDescriptions(writer);

            if (message == null)
                throw new WinPathResolverExit(writer.ToString());
            else
                throw new WinPathResolverException(writer.ToString());
        }
    }
}
