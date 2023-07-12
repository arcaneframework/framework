using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.IO;

namespace ArcaneInfra.WholeArchiveVCProj
{
    public static class XmlNodeExtender // méthode d'extension de XmlNode
    {
        public static XmlElement SelectOrCreateChild(this XmlNode node, string namespaceAlias, String name, XmlNamespaceManager nsMgr)
        {
            XmlNode child = node.SelectSingleNode(String.Format("{0}:{1}", namespaceAlias, name), nsMgr);
            if (child != null)
            {
                return (XmlElement)child;
            }
            else
            {
                XmlElement n = node.OwnerDocument.CreateElement(name, nsMgr.LookupNamespace(namespaceAlias));
                node.AppendChild(n);
                return n;
            }
        }
    }

    class Program
    {
        public enum VisualStudioKind { Undef, VCProj, VCXProj };

        static private void showHelp(Mono.Options.OptionSet opt_set) 
        {
            Console.WriteLine("Options :");
            opt_set.WriteOptionDescriptions(Console.Out);
            Environment.Exit(0);
        }

        static void Main(string[] args)
        {
            Mono.Options.OptionSet opt_set = new Mono.Options.OptionSet();

            bool show_help = false;
            string project = null;
            string path = null;
            string vstudio_name = null;
            bool remove_suffix = true;
            opt_set.Add("h|help|?", "help message", v => { show_help = true; });
            opt_set.Add("path=", "path of vcproj project", (string s) => path = s);
            opt_set.Add("project=", "vcproj project", (string s) => project = s);
            opt_set.Add("visual=", "Visual Studio name", (string s) => vstudio_name = s);
            opt_set.Add("dont-remove-suffix", "Don't remove suffix to name", v => { remove_suffix = false; });
            
			opt_set.Parse(args);
            
            if (show_help) showHelp(opt_set);
            
            if (project == null)
            {
                Console.Error.WriteLine("Error : no project specified");
                showHelp(opt_set);
            }

            if (path == null)
            {
                Console.Error.WriteLine("Error : no path specified");
                showHelp(opt_set);
            }

            if (vstudio_name == null)
            {
                Console.Error.WriteLine("Error : no visual studio name specified");
                showHelp(opt_set);
            }

            VisualStudioKind vstudio = VisualStudioKind.Undef;
            if (vstudio_name == "Visual Studio 9 2008 Win64")
            {
                vstudio = VisualStudioKind.VCProj;
            }
            else if (vstudio_name == "Visual Studio 10 Win64")
            {
                vstudio = VisualStudioKind.VCXProj;
            }
      		else if (vstudio_name == "Visual Studio 11 Win64")
      		{
      			vstudio = VisualStudioKind.VCXProj;
      		}
            else if (vstudio_name == "Visual Studio 12 Win64")
            {
              vstudio = VisualStudioKind.VCXProj;
            }
            else if (vstudio_name == "Visual Studio 12 2013 Win64")
            {
                vstudio = VisualStudioKind.VCXProj;
            }
            else
            {
                Console.Error.WriteLine("Unknown visual studio name '{0}'", vstudio_name);
                Environment.Exit(-1);                
            }

            string suffix = "";
            if (vstudio == VisualStudioKind.VCProj) 
                suffix = ".vcxproj";
            else if (vstudio == VisualStudioKind.VCXProj)
                suffix = ".vcxproj";

            string filename = Path.Combine(path, project);
            if(remove_suffix) {
                filename = Path.ChangeExtension(filename, suffix);
            } else {
                filename = filename + suffix;
            }
           
            Console.WriteLine("Apply Whole Archive patch on MS Visual studio project '{0}'", filename);

            XmlDocument vcprojXml = new XmlDocument();
            XmlNamespaceManager nsMgr = new XmlNamespaceManager(new NameTable()); // pour la gestion des namespaces (VS2010 et +)

            try
            {
                vcprojXml.Load(filename);
                vcprojXml.Save(filename + ".backup");
            }
            catch (System.IO.FileNotFoundException e)
            {
                Console.Error.WriteLine(e.Message);
                Environment.Exit(-1);
            }
            catch (XmlException e)
            {
                Console.Error.WriteLine(e.Message);
                Environment.Exit(-1);
            }

            if (vstudio == VisualStudioKind.VCProj)
            {
                XmlNodeList configs = vcprojXml.SelectNodes("VisualStudioProject/Configurations/Configuration/Tool[@Name='VCLinkerTool']");
                foreach (XmlElement config in configs)
                {
                    // TODO les exceoptions ...
                    config.Attributes["LinkLibraryDependencies"].Value = "true";
                    config.SetAttribute("UseLibraryDependencyInputs", "true");
                }
                Console.Error.WriteLine("WARNING: ignoring libcmt library not implemented");
            }
            else if (vstudio == VisualStudioKind.VCXProj)
            {
                const string namespaceURI = "http://schemas.microsoft.com/developer/msbuild/2003";
                nsMgr.AddNamespace("msb", namespaceURI);

                XmlNodeList project_configs = vcprojXml.SelectNodes(@"/msb:Project/msb:ItemDefinitionGroup/msb:ProjectReference", nsMgr);

                foreach (XmlNode config in project_configs)
                {
                    // TODO les exceoptions ...
                    config.SelectOrCreateChild("msb", "LinkLibraryDependencies", nsMgr).InnerText = "true";
                    config.SelectOrCreateChild("msb", "UseLibraryDependencyInputs", nsMgr).InnerText = "true";
                }

                XmlNodeList link_configs = vcprojXml.SelectNodes(@"/msb:Project/msb:ItemDefinitionGroup/msb:Link", nsMgr);

                foreach (XmlNode config in link_configs)
                {
                    // TODO les exceoptions ...
                    config.SelectOrCreateChild("msb", "IgnoreSpecificDefaultLibraries", nsMgr).InnerText = "libcmt.lib";
                }
            }

            try
            {
                vcprojXml.Save(filename);
            }
            catch (XmlException e)
            {
                Console.Error.WriteLine(e.Message);
                Environment.Exit(-1);
            }
        }
    }
}
